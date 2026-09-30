"""抓取并解析特朗普在美国政府道德办公室（OGE）的公开财务披露。

OGE 278-T 是有阈值、延迟和豁免的已报告交易流；年度 278e 也不是完整
经纪账户清单。本模块只保存官方文件中可验证的字段，不推断精确仓位或收益。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OgeDisclosureFetcher:
    """OGE 官方披露采集器。"""

    API_URL = "https://extapps2.oge.gov/201/Presiden.nsf/API.xsp/v2/rest"
    OFFICIAL_HOSTS = {"www.oge.gov", "oge.gov", "extapps2.oge.gov"}
    TRANSACTION_PATTERN = re.compile(
        r"(?P<asset>.{2,320}?)\s+"
        r"(?P<type>purchase|sale|exchange)\s+"
        r"(?P<date>\d{1,2}/\d{1,2}/(?:20)?\d{2})\s+"
        r"(?P<late>yes|no)\s+"
        r"\$?\s*(?P<minimum>[\d,]+)\s*[-–—]\s*\$?\s*(?P<maximum>[\d,]+)",
        re.IGNORECASE,
    )
    AMOUNT_PATTERN = re.compile(
        r"\$?\s*(?P<minimum>[\d,]+)\s*[-–—]\s*\$?\s*(?P<maximum>[\d,]+)"
    )
    HOLDING_PATTERN = re.compile(
        r"^\s*\d+\s+(?P<asset>.+?)\s+(?:Yes|No|N/A)\s+"
        r"(?P<value>\$[\d,]+\s*[-–—]\s*\$[\d,]+)"
        r"(?:\s+(?P<rest>.*))?$",
        re.IGNORECASE,
    )
    SECURITY_TERMS = (
        "stock",
        "common",
        "class a",
        "class b",
        "shares",
        "etf",
        "fund",
        "trust",
        "bond",
        "note",
        "corp",
        "corporation",
        " inc",
        " plc",
        " l.p",
        " lp",
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.oge_config = config.get("oge_disclosure", {})
        fetcher_config = config.get("fetcher", {})
        self.enabled = self.oge_config.get("enabled", False)
        self.timeout = self.oge_config.get(
            "timeout_seconds", fetcher_config.get("timeout_seconds", 30)
        )
        self.retry_attempts = self.oge_config.get(
            "retry_attempts", fetcher_config.get("retry_attempts", 2)
        )
        self.retry_delay = self.oge_config.get(
            "retry_delay_seconds", fetcher_config.get("retry_delay_seconds", 3)
        )
        self.lookback_days = int(self.oge_config.get("lookback_days", 180))
        self.max_filings = int(self.oge_config.get("max_filings", 12))
        self.max_transactions = int(
            self.oge_config.get("max_transactions", 3000)
        )
        self.max_holdings = int(self.oge_config.get("max_holdings", 500))
        self.max_pdf_pages = int(self.oge_config.get("max_pdf_pages", 500))
        self.extract_pdf_tables = bool(
            self.oge_config.get("extract_pdf_tables", False)
        )
        self.ocr_enabled = bool(
            self.oge_config.get("ocr_enabled", True)
        ) and sys.platform == "darwin"
        self.ocr_resolution = int(
            self.oge_config.get("ocr_resolution", 200)
        )
        self.ocr_max_pages = int(
            self.oge_config.get("ocr_max_pages_per_filing", 150)
        )
        self.ocr_min_text_chars = int(
            self.oge_config.get("ocr_min_text_chars", 80)
        )
        self.cache_extracted_pages = bool(
            self.oge_config.get("cache_extracted_pages", True)
        )
        self.cache_dir = Path(
            self.oge_config.get("pdf_cache_dir", "data/oge_pdf_cache")
        )
        self.snapshot_path = self.cache_dir / "latest_disclosure.json"
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": fetcher_config.get("user_agent", "NewsAnalyzer/1.0")}
        )

    def fetch_all_disclosures(
        self, articles: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """抓取最近披露；失败时回退到最近一次成功快照。"""
        result = self._empty_result()
        if not self.enabled:
            logger.info("特朗普 OGE 财务披露抓取未启用")
            return result

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            filings = self._discover_filings()
            if not filings:
                raise RuntimeError("OGE 官方索引未发现特朗普披露文件")
            result["filings"] = filings
            self._parse_filings(result)
            result["background_news"] = self._filter_background_news(articles or [])
            result["data_as_of"] = self._latest_date(
                [
                    item.get("transaction_date")
                    for item in result["transactions"]
                ]
                + [item.get("received_date") for item in filings]
            )
            self._save_snapshot(result)
        except Exception as exc:
            logger.warning("抓取特朗普 OGE 财务披露失败: %s", exc)
            cached = self._load_snapshot()
            if cached:
                result = cached
                result["stale_cache"] = True
                result.setdefault("errors", []).append(
                    f"OGE 官方源本轮不可用，使用截至 {result.get('data_as_of') or '未知'} "
                    f"的缓存: {exc}"
                )
            else:
                result["errors"].append(f"OGE 官方披露: {exc}")
                result["background_news"] = self._filter_background_news(
                    articles or []
                )

        logger.info(
            "特朗普 OGE 披露完成：文件 %d 份，交易 %d 笔，年度持仓基线 %d 项",
            len(result.get("filings", [])),
            len(result.get("transactions", [])),
            len(result.get("holdings", [])),
        )
        return result

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data_as_of": None,
            "subject": {
                "name": "Donald J. Trump",
                "title": "President of the United States",
            },
            "filings": [],
            "transactions": [],
            "holdings": [],
            "background_news": [],
            "errors": [],
            "stale_cache": False,
            "scope_note": (
                "278-T仅为超过1,000美元且不属于豁免范围的已报告交易；"
                "年度278e为法定阈值和豁免后的持仓基线，均非实时或完整投资组合。"
            ),
        }

    def _discover_filings(self) -> List[Dict[str, Any]]:
        """从 OGE 官方 DataTables API 发现文件，并合并显式官方 URL。"""
        api_url = self.oge_config.get("api_url", self.API_URL)
        scan_limit = max(
            self.max_filings, int(self.oge_config.get("index_scan_limit", 2000))
        )
        data = self._request_json(
            api_url, {"start": 0, "length": scan_limit}
        )
        accepted_names = {
            self._normalize_name(value)
            for value in self.oge_config.get(
                "subject_names", ["Trump, Donald J", "Donald J. Trump"]
            )
        }
        cutoff = datetime.now(timezone.utc).date() - timedelta(
            days=self.lookback_days
        )
        filings: List[Dict[str, Any]] = []
        for row in data.get("data", []):
            if self._normalize_name(row.get("name", "")) not in accepted_names:
                continue
            received_date = str(row.get("docDate", ""))[:10]
            parsed_received = self._parse_iso_date(received_date)
            form_type, report_year, pdf_url = self._parse_api_type(
                row.get("type", "")
            )
            if not pdf_url or form_type not in {"278-T", "278e Annual"}:
                continue
            if form_type == "278-T" and parsed_received and parsed_received < cutoff:
                continue
            filings.append(
                {
                    "filing_id": hashlib.sha256(
                        pdf_url.encode("utf-8")
                    ).hexdigest()[:16],
                    "form_type": form_type,
                    "report_year": report_year,
                    "received_date": received_date or None,
                    "name": row.get("name", ""),
                    "title": row.get("title", ""),
                    "agency": row.get("agency", ""),
                    "source": "U.S. Office of Government Ethics",
                    "source_url": pdf_url,
                    "is_official": True,
                    "parse_status": "pending",
                    "amended": bool(row.get("amended")),
                }
            )

        for value in self.oge_config.get("known_filing_urls", []):
            if not value or not self._is_official_url(value):
                continue
            form_type = "278-T" if "278-t" in value.lower() else "278e Annual"
            filings.append(
                {
                    "filing_id": hashlib.sha256(value.encode("utf-8")).hexdigest()[
                        :16
                    ],
                    "form_type": form_type,
                    "report_year": None,
                    "received_date": None,
                    "name": "Donald J. Trump",
                    "title": "President of the United States",
                    "agency": "Executive Office of the President",
                    "source": "U.S. Office of Government Ethics",
                    "source_url": value,
                    "is_official": True,
                    "parse_status": "pending",
                    "amended": False,
                }
            )

        deduplicated: Dict[str, Dict[str, Any]] = {}
        for filing in filings:
            deduplicated[filing["source_url"]] = filing
        ordered = sorted(
            deduplicated.values(),
            key=lambda item: item.get("received_date") or "",
            reverse=True,
        )
        transactions = [
            item for item in ordered if item["form_type"] == "278-T"
        ][: self.max_filings]
        annual = next(
            (item for item in ordered if item["form_type"] == "278e Annual"),
            None,
        )
        if annual:
            transactions.append(annual)
        return transactions

    def _parse_filings(self, result: Dict[str, Any]) -> None:
        seen_transactions = set()
        seen_holdings = set()
        for filing in result["filings"]:
            try:
                pdf_bytes, pdf_path = self._download_pdf(filing["source_url"])
                filing["sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
                filing["cached_pdf"] = str(pdf_path)
                pages, total_pages = self._extract_pdf_pages(
                    pdf_bytes, allow_ocr=filing["form_type"] == "278-T"
                )
                filing["pages_total"] = total_pages
                filing["pages_parsed"] = len(pages)
                filing["ocr_pages"] = sum(
                    1 for page in pages if page.get("ocr")
                )
                fully_parsed = bool(pages) and len(pages) == total_pages
                if filing["ocr_pages"]:
                    filing["parse_status"] = (
                        "parsed_with_ocr"
                        if fully_parsed
                        else "partially_parsed_with_ocr"
                    )
                else:
                    filing["parse_status"] = (
                        "parsed" if fully_parsed else "partially_parsed"
                    )
                if filing["form_type"] == "278-T":
                    parsed = self._parse_transactions(pages, filing)
                    if not parsed:
                        filing["parse_status"] = "metadata_only"
                        filing["parse_note"] = (
                            "PDF未提取到可靠交易表格，可能为扫描页，待人工复核"
                        )
                    for item in parsed:
                        if (
                            len(result["transactions"])
                            >= self.max_transactions
                        ):
                            break
                        key = (
                            item["asset_description"].lower(),
                            item["transaction_type"],
                            item["transaction_date"],
                            item["amount_min_usd"],
                            item["amount_max_usd"],
                        )
                        if key in seen_transactions:
                            continue
                        seen_transactions.add(key)
                        result["transactions"].append(item)
                else:
                    parsed = self._parse_holdings(pages, filing)
                    if not parsed:
                        filing["parse_status"] = "metadata_only"
                        filing["parse_note"] = (
                            "年度PDF未提取到可靠证券持仓行，待人工复核"
                        )
                    for item in parsed:
                        if len(result["holdings"]) >= self.max_holdings:
                            break
                        key = (
                            item["asset_description"].lower(),
                            item["value_range"],
                        )
                        if key in seen_holdings:
                            continue
                        seen_holdings.add(key)
                        result["holdings"].append(item)
                if not pages:
                    filing["parse_status"] = "metadata_only"
                    filing["parse_note"] = "PDF无可靠文本层，待人工复核"
            except Exception as exc:
                filing["parse_status"] = "failed"
                filing["parse_note"] = str(exc)
                result["errors"].append(
                    f"{filing['form_type']} {filing.get('received_date') or ''}: {exc}"
                )

    def _download_pdf(self, url: str) -> Tuple[bytes, Path]:
        if not self._is_official_url(url):
            raise RuntimeError("拒绝下载非 OGE 官方域名文件")
        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".pdf"
        path = self.cache_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return path.read_bytes(), path
        content = self._request_bytes(url)
        if not content.startswith(b"%PDF"):
            raise RuntimeError("OGE 文件不是有效 PDF")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return content, path

    def _extract_pdf_pages(
        self, content: bytes, allow_ocr: bool = False
    ) -> Tuple[List[Dict[str, Any]], int]:
        text_cache_path = self._page_text_cache_path(content, allow_ocr)
        if self.cache_extracted_pages:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_extracted_pages and text_cache_path.exists():
            try:
                cached = json.loads(text_cache_path.read_text(encoding="utf-8"))
                return cached["pages"], int(cached["total_pages"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("读取 OGE PDF 文本缓存失败: %s", exc)
        pages: List[Dict[str, Any]] = []
        ocr_pages = 0
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            total_pages = len(pdf.pages)
            for page_number, page in enumerate(
                pdf.pages[: self.max_pdf_pages], start=1
            ):
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                used_ocr = False
                if (
                    allow_ocr
                    and self.ocr_enabled
                    and len(text.strip()) < self.ocr_min_text_chars
                    and ocr_pages < self.ocr_max_pages
                ):
                    try:
                        ocr_text = self._ocr_pdf_page(page)
                        if len(ocr_text.strip()) > len(text.strip()):
                            text = ocr_text
                            used_ocr = True
                        ocr_pages += 1
                    except Exception as exc:
                        logger.warning(
                            "OGE PDF 第 %d 页 OCR 失败: %s",
                            page_number,
                            exc,
                        )
                tables = (
                    page.extract_tables()
                    if text and self.extract_pdf_tables
                    else []
                )
                if text.strip() or tables:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                            "tables": tables,
                            "ocr": used_ocr,
                        }
                    )
        if self.cache_extracted_pages:
            try:
                temporary = text_cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {"total_pages": total_pages, "pages": pages},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(text_cache_path)
            except OSError as exc:
                logger.warning("保存 OGE PDF 文本缓存失败: %s", exc)
        return pages, total_pages

    def _page_text_cache_path(
        self, content: bytes, allow_ocr: bool
    ) -> Path:
        digest = hashlib.sha256(content).hexdigest()
        mode = (
            f"ocr{int(self.ocr_enabled and allow_ocr)}"
            f"-r{self.ocr_resolution}-p{self.max_pdf_pages}"
        )
        return self.cache_dir / f"{digest}.{mode}.pages-v3.json"

    def _ocr_pdf_page(self, page: Any) -> str:
        """使用 macOS Vision 对扫描页做本地 OCR；不可用时由调用方降级。"""
        try:
            from Foundation import NSURL
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )
        except ImportError as exc:
            raise RuntimeError("macOS Vision OCR 依赖未安装") from exc

        image = page.to_image(
            resolution=self.ocr_resolution, antialias=True
        ).original
        text = self._recognize_image_with_vision(
            image,
            NSURL,
            VNImageRequestHandler,
            VNRecognizeTextRequest,
            VNRequestTextRecognitionLevelAccurate,
        )
        best_text = text
        best_score = self._ocr_transaction_score(text)
        if (
            self.oge_config.get("ocr_try_rotate_180", True)
            and best_score < 5
        ):
            for angle in (180, 90, 270):
                rotated_text = self._recognize_image_with_vision(
                    image.rotate(angle, expand=True),
                    NSURL,
                    VNImageRequestHandler,
                    VNRecognizeTextRequest,
                    VNRequestTextRecognitionLevelAccurate,
                )
                score = self._ocr_transaction_score(rotated_text)
                if score > best_score:
                    best_text = rotated_text
                    best_score = score
                if best_score >= 5:
                    break
        return best_text

    @classmethod
    def _ocr_transaction_score(cls, text: str) -> int:
        return sum(
            1
            for line in (text or "").splitlines()
            if cls.TRANSACTION_PATTERN.search(" ".join(line.split()))
        )

    @staticmethod
    def _recognize_image_with_vision(
        image: Any,
        ns_url: Any,
        handler_class: Any,
        request_class: Any,
        accurate_level: Any,
    ) -> str:
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temporary:
                temporary_path = temporary.name
            image.save(temporary_path, format="PNG")
            request = request_class.alloc().init()
            request.setRecognitionLevel_(accurate_level)
            request.setRecognitionLanguages_(["en-US"])
            request.setUsesLanguageCorrection_(False)
            handler = handler_class.alloc().initWithURL_options_(
                ns_url.fileURLWithPath_(temporary_path), {}
            )
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise RuntimeError(f"Vision OCR 请求失败: {error}")
            observations = []
            for observation in request.results() or []:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                box = observation.boundingBox()
                observations.append(
                    (
                        float(box.origin.y + box.size.height / 2),
                        float(box.origin.x),
                        str(candidates[0].string()),
                    )
                )
            observations.sort(key=lambda item: (-item[0], item[1]))
            lines: List[List[Tuple[float, str]]] = []
            line_y_values: List[float] = []
            for y_value, x_value, text in observations:
                target = None
                for index, existing_y in enumerate(line_y_values):
                    if abs(existing_y - y_value) <= 0.012:
                        target = index
                        break
                if target is None:
                    line_y_values.append(y_value)
                    lines.append([])
                    target = len(lines) - 1
                lines[target].append((x_value, text))
            return "\n".join(
                " ".join(text for _, text in sorted(line))
                for line in lines
                if line
            )
        finally:
            if temporary_path:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _parse_transactions(
        self, pages: List[Dict[str, Any]], filing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for page in pages:
            candidates = list(self._table_lines(page.get("tables", [])))
            candidates.extend(
                " ".join(line.split())
                for line in page.get("text", "").splitlines()
                if line.strip()
            )
            for line in candidates:
                match = self.TRANSACTION_PATTERN.search(line)
                if not match:
                    continue
                asset = self._clean_asset_description(match.group("asset"))
                if not asset:
                    continue
                transaction_date = self._normalize_us_date(match.group("date"))
                if not transaction_date or not self._is_plausible_transaction_date(
                    transaction_date, filing.get("received_date")
                ):
                    continue
                amount_min = self._to_int(match.group("minimum"))
                amount_max = self._to_int(match.group("maximum"))
                transaction_type = match.group("type").lower()
                items.append(
                    {
                        "asset_description": asset,
                        "ticker": None,
                        "transaction_type": {
                            "purchase": "Purchase",
                            "sale": "Sale",
                            "exchange": "Exchange",
                        }[transaction_type],
                        "transaction_date": transaction_date,
                        "amount_range": f"${amount_min:,} - ${amount_max:,}",
                        "amount_min_usd": amount_min,
                        "amount_max_usd": amount_max,
                        "late_filing": match.group("late").lower() == "yes",
                        "discretionary_order": (
                            "discretionary order" in line.lower()
                        ),
                        "owner": "未在解析字段中可靠识别",
                        "page": page["page"],
                        "filing_id": filing["filing_id"],
                        "source": "OGE Form 278-T",
                        "source_url": filing["source_url"],
                        "is_official": True,
                        "parse_confidence": "high",
                    }
                )
        return items

    @classmethod
    def _is_plausible_transaction_date(
        cls, transaction_date: str, received_date: Optional[str]
    ) -> bool:
        parsed_transaction = cls._parse_iso_date(transaction_date)
        if not parsed_transaction:
            return False
        upper_bound = datetime.now(timezone.utc).date() + timedelta(days=1)
        if received_date:
            parsed_received = cls._parse_iso_date(received_date)
            if parsed_received:
                upper_bound = min(upper_bound, parsed_received + timedelta(days=1))
                if parsed_transaction < parsed_received - timedelta(days=730):
                    return False
        return parsed_transaction <= upper_bound

    def _parse_holdings(
        self, pages: List[Dict[str, Any]], filing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for page in pages:
            for line in self._holding_lines(page.get("text", "")):
                match = self.HOLDING_PATTERN.search(line)
                if not match:
                    continue
                description = self._clean_asset_description(match.group("asset"))
                value_range = " ".join(match.group("value").split())
                if not description or not self._is_security_asset(description):
                    continue
                rest = " ".join((match.group("rest") or "").split())
                items.append(
                    self._holding_item(
                        description,
                        value_range,
                        self._holding_income_type(rest),
                        self._holding_income_amount(rest),
                        page["page"],
                        filing,
                        "medium",
                    )
                )
            for table in page.get("tables", []):
                if not table:
                    continue
                header_index, mapping = self._holding_header(table)
                if header_index is None:
                    continue
                for row in table[header_index + 1 :]:
                    cells = [self._cell(value) for value in row]
                    description = self._mapped_cell(cells, mapping, "description")
                    value_range = self._mapped_cell(cells, mapping, "value")
                    if not description or not value_range:
                        continue
                    if not self.AMOUNT_PATTERN.search(value_range):
                        continue
                    if not self._is_security_asset(description):
                        continue
                    items.append(
                        self._holding_item(
                            description,
                            value_range,
                            self._mapped_cell(cells, mapping, "income_type")
                            or "未可靠识别",
                            self._mapped_cell(cells, mapping, "income_amount")
                            or "未可靠识别",
                            page["page"],
                            filing,
                            "medium",
                        )
                    )
        return items

    def _holding_item(
        self,
        description: str,
        value_range: str,
        income_type: str,
        income_amount: str,
        page_number: int,
        filing: Dict[str, Any],
        confidence: str,
    ) -> Dict[str, Any]:
        amount_match = self.AMOUNT_PATTERN.search(value_range)
        value_min = (
            self._to_int(amount_match.group("minimum"))
            if amount_match
            else None
        )
        value_max = (
            self._to_int(amount_match.group("maximum"))
            if amount_match
            else None
        )
        return {
            "asset_description": description,
            "asset_type": self._infer_asset_type(description),
            "value_range": value_range,
            "value_min_usd": value_min,
            "value_max_usd": value_max,
            "income_type": income_type,
            "income_amount": income_amount,
            "owner": "未在解析字段中可靠识别",
            "as_of_date": (
                str(filing.get("report_year"))
                if filing.get("report_year")
                else "年度报告期末"
            ),
            "page": page_number,
            "filing_id": filing["filing_id"],
            "source": "OGE Form 278e Annual",
            "source_url": filing["source_url"],
            "is_official": True,
            "parse_confidence": confidence,
        }

    @classmethod
    def _holding_income_amount(cls, value: str) -> str:
        ranges = list(cls.AMOUNT_PATTERN.finditer(value or ""))
        if not ranges:
            return "未可靠识别"
        match = ranges[-1]
        return f"${match.group('minimum')} - ${match.group('maximum')}"

    @staticmethod
    def _holding_income_type(value: str) -> str:
        upper = (value or "").upper()
        for name in (
            "DIVIDEND/CAPITAL GAINS",
            "CAPITAL GAINS",
            "DIVIDEND",
            "INTEREST",
            "RENT OR ROYALTIES",
        ):
            if name in upper:
                return name.title()
        if "NONE" in upper:
            return "None"
        return "未可靠识别"

    @staticmethod
    def _holding_header(
        table: List[List[Any]],
    ) -> Tuple[Optional[int], Dict[str, int]]:
        for index, row in enumerate(table[:5]):
            cells = [
                " ".join(str(value or "").lower().split()) for value in row
            ]
            mapping: Dict[str, int] = {}
            for cell_index, cell in enumerate(cells):
                if "description" in cell or "asset" in cell:
                    mapping.setdefault("description", cell_index)
                if "value" in cell:
                    mapping.setdefault("value", cell_index)
                if "income type" in cell:
                    mapping.setdefault("income_type", cell_index)
                if "income amount" in cell:
                    mapping.setdefault("income_amount", cell_index)
            if "description" in mapping and "value" in mapping:
                return index, mapping
        return None, {}

    @staticmethod
    def _mapped_cell(
        cells: List[str], mapping: Dict[str, int], key: str
    ) -> str:
        index = mapping.get(key)
        if index is None or index >= len(cells):
            return ""
        return cells[index]

    @classmethod
    def _table_lines(cls, tables: Iterable[List[List[Any]]]) -> Iterable[str]:
        for table in tables:
            for row in table or []:
                line = " ".join(cls._cell(value) for value in row if value)
                if line:
                    yield line

    @staticmethod
    def _combined_lines(text: str) -> Iterable[str]:
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            yield line
            if index + 1 < len(lines):
                yield f"{line} {lines[index + 1]}"
            if index + 2 < len(lines):
                yield f"{line} {lines[index + 1]} {lines[index + 2]}"

    @staticmethod
    def _holding_lines(text: str) -> Iterable[str]:
        """按年度披露的行号边界拼接换行，避免把相邻资产合并。"""
        groups: List[List[str]] = []
        current: List[str] = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            if re.match(r"^\d+\s+\S", line):
                if current:
                    groups.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            groups.append(current)
        for group in groups:
            yield " ".join(group)

    @staticmethod
    def _parse_api_type(value: str) -> Tuple[str, Optional[int], Optional[str]]:
        soup = BeautifulSoup(value or "", "html.parser")
        text = " ".join(soup.get_text(" ", strip=True).split())
        anchor = soup.find("a", href=True)
        url = anchor["href"] if anchor else None
        lower = text.lower()
        form_type = "278-T" if "transaction" in lower else "278e Annual"
        year_match = re.search(r"\((20\d{2})\)", text)
        report_year = int(year_match.group(1)) if year_match else None
        return form_type, report_year, url

    def _filter_background_news(self, articles: List[Any]) -> List[Dict[str, Any]]:
        trusted = {
            value.lower()
            for value in self.oge_config.get(
                "trusted_background_sources",
                ["Reuters", "Associated Press", "CNBC", "Financial Times", "BBC"],
            )
        }
        keywords = [
            value.lower()
            for value in self.oge_config.get(
                "background_news_keywords",
                [
                    "trump",
                    "278-t",
                    "financial disclosure",
                    "stock trade",
                    "securities trade",
                ],
            )
        ]
        limit = int(self.oge_config.get("max_background_articles", 6))
        items = []
        for article in articles:
            source = str(getattr(article, "source", ""))
            haystack = " ".join(
                [
                    str(getattr(article, "title", "")),
                    str(getattr(article, "description", "")),
                ]
            ).lower()
            if source.lower() not in trusted:
                continue
            if "trump" not in haystack or not any(
                keyword in haystack for keyword in keywords if keyword != "trump"
            ):
                continue
            published = getattr(article, "published_at", None)
            items.append(
                {
                    "title": getattr(article, "title", ""),
                    "source": source,
                    "published_at": (
                        published.isoformat()
                        if hasattr(published, "isoformat")
                        else str(published or "")
                    ),
                    "url": getattr(article, "url", ""),
                    "note": "可信媒体背景，不得替代 OGE 官方披露字段",
                }
            )
        return items[:limit]

    def _request_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = self._request(url, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("OGE 索引返回非 JSON 内容") from exc

    def _request_bytes(self, url: str) -> bytes:
        return self._request(url).content

    def _request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        if not self._is_official_url(url):
            raise RuntimeError("拒绝访问非 OGE 官方域名")
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout
                )
                if response.status_code == 429:
                    raise RuntimeError("OGE HTTP 429，本轮停止该数据源请求")
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError(f"OGE 请求失败: {exc}") from exc
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("OGE 请求失败")

    def _save_snapshot(self, result: Dict[str, Any]) -> None:
        temporary = self.snapshot_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.snapshot_path)

    def _load_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.snapshot_path.exists():
                return None
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("读取 OGE 披露缓存失败: %s", exc)
            return None

    @classmethod
    def _is_official_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in cls.OFFICIAL_HOSTS

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"[^a-z]", "", (value or "").lower())

    @staticmethod
    def _parse_iso_date(value: str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_us_date(value: str) -> Optional[str]:
        for pattern in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(value, pattern).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _to_int(value: str) -> int:
        return int(str(value).replace(",", ""))

    @staticmethod
    def _cell(value: Any) -> str:
        return " ".join(str(value or "").replace("\n", " ").split())

    @staticmethod
    def _clean_asset_description(value: str) -> str:
        cleaned = re.sub(r"^\s*\d+\s*[-.)]?\s*", "", value or "")
        cleaned = re.sub(
            r"^(transactions?|asset description)\s*", "", cleaned, flags=re.I
        )
        return " ".join(cleaned.split()).strip(" -|")

    @classmethod
    def _is_security_asset(cls, value: str) -> bool:
        lower = f" {value.lower()} "
        return any(term in lower for term in cls.SECURITY_TERMS)

    @classmethod
    def _infer_asset_type(cls, value: str) -> str:
        lower = value.lower()
        if (
            "bond" in lower
            or "note" in lower
            or " deb" in lower
            or " nts" in lower
            or " due " in lower
            or re.search(r"\d\.\d{2,3}%", lower)
        ):
            return "Bond/Note"
        if "etf" in lower or "fund" in lower:
            return "Fund/ETF"
        if "trust" in lower:
            return "Trust"
        return "Stock/Other security"

    @staticmethod
    def _latest_date(values: Iterable[Optional[str]]) -> Optional[str]:
        dates = [value for value in values if value]
        return max(dates) if dates else None

