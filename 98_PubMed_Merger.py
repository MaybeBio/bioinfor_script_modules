# 参考我的pypaperflow模块，其中为pubmed设计的数据提取的原始合并模块

"""
Paper Merger Module for pyPaperFlow

This module provides functionality to merge PubMed paper metadata and content
into unified formats optimized for AI analysis.

⚠️ For Pubmed papers ONLY
"""


import os
import json
import csv
from typing import *
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import yaml

# md for LLM, so we can split into sections
# for a typical pubmed paper, 
DEFAULT_LLM_SECTIONS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
]

# ⚠️
def _normalize_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())

# ⚠️ 
def _safe_copy(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return obj


def _timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# ⚠️
class MergeMode(Enum):
    """Merge modes."""

    METADATA_ONLY = "metadata_only"
    METADATA_CONTENT = "metadata_content"

# ⚠️
class OutputFormat(Enum):
    """Supported output formats."""

    MARKDOWN = "md"
    JSONL = "jsonl"
    PLAIN_TEXT = "txt"

# ⚠️
class OutputProfile(Enum):
    """Output profile for downstream usage."""

    LLM = "llm"
    ANALYSIS = "analysis"


# ⚠️ 
class TextSourcePriority(Enum):
    """Priority when both parsed JSON and markdown are available."""

    PARSED_JSON_FIRST = "parsed-json-first"
    PARSED_MD_FIRST = "parsed-md-first"


@dataclass
class MergeConfig:
    """Configuration for merge behavior."""

    mode: MergeMode = MergeMode.METADATA_CONTENT  # ⚠️ 
    output_format: OutputFormat = OutputFormat.MARKDOWN  # ⚠️
    output_profile: OutputProfile = OutputProfile.ANALYSIS # ⚠️
    include_sections: Optional[List[str]] = None  # ⚠️ content中需要包括的部分 
    include_metadata_fields: Optional[List[str]] = None  #  metadata中需要包括的部分 
    include_links_in_llm: bool = False  # ⚠️ 
    text_source_priority: TextSourcePriority = TextSourcePriority.PARSED_JSON_FIRST # ⚠️ 


class PaperMerger:
    """
    A class to merge PubMed paper data into unified formats.

    Minimal and robust merge workflow:
    - Input source: directory scan or explicit PMID file
    - Merge mode:  metadata only, or metadata + full text  
    - Output format: Markdown / JSONL / plain text 
    - Output profile: LLM-oriented or analysis-oriented
    """

    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize the PaperMerger.

        Args:
            config: MergeConfig object with merge settings
        """
        self.config = config or MergeConfig()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'failed_pmids': []
        }

    def merge_from_directory(self, paper_dir: str, output_path: str) -> Dict:
        """
        Merge papers from a directory structure.

        Args:
            paper_dir: Directory containing papers (pubmed/year/pmid/files)
            output_path: Path to output file

        Returns:
            Dictionary with merge statistics
        """
        if not os.path.exists(paper_dir):
            raise ValueError(f"Directory does not exist: {paper_dir}")

        papers = self._collect_papers_from_directory(paper_dir)

        if not papers:
            print("Warning: No papers found in the directory.")
            return self.stats

        return self._merge_papers(papers, output_path)

    def merge_json_from_directory(
        self,
        paper_dir: str,
        output_json: str,
        pmid_file: Optional[str] = None,
        jsonl: bool = False,
    ) -> Dict:
        """Merge meta+content into a single JSON array (or JSONL) file.

        This is the core "merge JSON" stage that upstream tools should call.
        """
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'failed_pmids': []
        }

        if pmid_file:
            papers = self._collect_papers_from_pmids(paper_dir, self._read_pmids_from_file(pmid_file))
        else:
            papers = self._collect_papers_from_directory(paper_dir)

        if not papers:
            print("Warning: No papers found for JSON merge.")
            return self.stats

        source_name = os.path.splitext(os.path.basename(os.path.normpath(paper_dir)))[0] or "pubmed"
        if pmid_file:
            source_name = os.path.splitext(os.path.basename(pmid_file))[0] or source_name

        output_json = self._resolve_merged_output_path(output_json, source_name, jsonl=jsonl)

        # ensure output dir
        outdir = os.path.dirname(output_json)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        try:
            if jsonl:
                with open(output_json, 'w', encoding='utf-8') as handle:
                    for paper in papers:
                        handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
            else:
                with open(output_json, 'w', encoding='utf-8') as handle:
                    json.dump(papers, handle, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to write merged JSON: {e}")

        self.stats['total_processed'] = len(papers)
        self.stats['successful'] = len(papers)
        self.stats['output_path'] = output_json
        return self.stats

    def load_merged_json(self, path: str) -> List[Dict[str, Any]]:
        """Load a merged JSON array or JSONL file and return list of paper dicts."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    return data
        except Exception:
            # fallback to JSONL
            papers: List[Dict[str, Any]] = []
            with open(path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        papers.append(json.loads(line))
                    except Exception:
                        continue
            return papers
        return []

    def export_md_from_merged_json(
        self,
        merged_json: str,
        output_md: str,
        yaml_cfg: Optional[str] = None,
        pmid_file: Optional[str] = None,
    ) -> Dict:
        """Export Markdown from merged JSON using optional YAML config.

        YAML schema (minimal):
        metadata_fields: ["identity.title", "source.pub_date"]
        content_sections: ["abstract", "methods"]
        """
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'failed_pmids': []
        }

        papers = self.load_merged_json(merged_json)

        if pmid_file:
            pmids = set(self._read_pmids_from_file(pmid_file))
            papers = [p for p in papers if str(p.get('pmid', '')) in pmids]

        cfg: Dict[str, Any] = {}
        if yaml_cfg:
            try:
                with open(yaml_cfg, 'r', encoding='utf-8') as yf:
                    cfg = yaml.safe_load(yf) or {}
            except Exception as e:
                raise RuntimeError(f"Failed to load YAML config: {e}")

        metadata_fields = cfg.get('metadata_fields', ['identity', 'source'])
        content_sections = cfg.get('content_sections', None)

        # write markdown
        outdir = os.path.dirname(output_md)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        with open(output_md, 'w', encoding='utf-8') as f:
            f.write("# Merged Literature Corpus\n\n")
            f.write(f"Total papers: {len(papers)}\n\n")

            for idx, paper in enumerate(papers, 1):
                pmid = str(paper.get('pmid', '')).strip() or 'N/A'
                title = self._scalar((paper.get('identity') or {}).get('title', 'N/A'))
                f.write(f"## PMID {pmid} - {title}\n\n")
                # metadata selection
                for field in metadata_fields:
                    value = self._get_by_path(paper, field) if isinstance(field, str) else None
                    if value is None:
                        continue
                    # flatten lists
                    if isinstance(value, list):
                        value = ', '.join(str(x) for x in value)
                    f.write(f"**{field}:** {value}  \n")
                f.write("\n")

                # content sections
                text_sections = paper.get('text_sections') or []
                if not content_sections:
                    content_sections = DEFAULT_LLM_SECTIONS

                content_present = False
                if content_sections:
                    for sec in content_sections:
                        norm = _normalize_name(sec)
                        sec_text = None
                        for s in text_sections:
                            if _normalize_name(s.get('title', '')) == norm:
                                sec_text = str(s.get('content', '')).strip()
                                break
                        f.write(f"### {sec.title()}\n\n")
                        if sec_text:
                            f.write(sec_text + "\n\n")
                            content_present = True
                        else:
                            f.write("None\n\n")
                else:
                    f.write("### Abstract\n\n")
                    abstract = (paper.get('content') or {}).get('abstract')
                    if abstract:
                        f.write(str(abstract).strip() + "\n\n")
                        content_present = True
                    else:
                        f.write("None\n\n")

                    full_text = paper.get('full_text')
                    f.write("### Full Text\n\n")
                    if full_text and str(full_text).strip():
                        f.write(str(full_text).strip() + "\n\n")
                        content_present = True
                    else:
                        f.write("None\n\n")

                if idx < len(papers):
                    f.write("\n<!-- PAPER_BREAK -->\n\n---\n\n")

        self.stats['total_processed'] = len(papers)
        self.stats['successful'] = len(papers)
        self.stats['output_path'] = output_md
        return self.stats

    def merge_from_file_list(
        self,
        paper_dir: str,
        pmid_file: str,
        output_path: str
    ) -> Dict:
        """
        Merge papers specified in a file.

        Args:
            paper_dir: Base directory containing papers
            pmid_file: Path to file containing PMIDs (one per line or CSV)
            output_path: Path to output file

        Returns:
            Dictionary with merge statistics
        """
        pmids = self._read_pmids_from_file(pmid_file)

        if not pmids:
            raise ValueError(f"No PMIDs found in file: {pmid_file}")

        papers = self._collect_papers_from_pmids(paper_dir, pmids)

        if not papers:
            print("Warning: No matching papers found.")
            return self.stats

        return self._merge_papers(papers, output_path)

    def _collect_papers_from_directory(self, paper_dir: str) -> List[Dict]:
        """Collect paper data from directory structure."""
        papers = []

        if not os.path.exists(paper_dir):
            print(f"Error: Directory does not exist {paper_dir}")
            return papers

        for pmid, pmid_dir in self._iter_pmid_directories(paper_dir):
            paper_data = self._load_paper_data(pmid_dir, pmid)
            if paper_data:
                self._write_paper_json(pmid_dir, paper_data)
                papers.append(paper_data)

        self.stats['total_processed'] = len(papers)
        return papers

    def _collect_papers_from_pmids(
        self,
        paper_dir: str,
        pmids: List[str]
    ) -> List[Dict]:
        """Collect paper data for specific PMIDs."""
        papers = []

        # Build an index once to avoid repeated directory scans.
        pmid_dir_map = self._index_pmid_directories(paper_dir)

        for pmid in pmids:
            pmid = str(pmid).strip()
            if not pmid:
                continue

            # Search in year subdirectories
            pmid_dir = pmid_dir_map.get(pmid)
            if pmid_dir:
                paper_data = self._load_paper_data(pmid_dir, pmid)
                if paper_data:
                    self._write_paper_json(pmid_dir, paper_data)
                    papers.append(paper_data)
                    continue

            self.stats['skipped'] += 1
            print(f"Warning: PMID {pmid} not found in directory")

        self.stats['total_processed'] = len(papers)
        return papers

    def _index_pmid_directories(self, paper_dir: str) -> Dict[str, str]:
        """Build PMID -> directory map from year/pmid folder layout."""
        pmid_dir_map: Dict[str, str] = {}

        for pmid, pmid_dir in self._iter_pmid_directories(paper_dir):
            pmid_dir_map[str(pmid)] = pmid_dir

        return pmid_dir_map

    def _iter_pmid_directories(self, paper_dir: str) -> List[tuple[str, str]]:
        """Yield (pmid, directory) pairs for root/pubmed/year/pmid layouts."""
        pairs: List[tuple[str, str]] = []

        if not os.path.isdir(paper_dir):
            return pairs

        entries = sorted(os.listdir(paper_dir))

        # Case 1: paper_dir directly contains PMID folders.
        direct_pmid_dirs: List[tuple[str, str]] = []
        for entry in entries:
            entry_path = os.path.join(paper_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            # accept {pmid}_meta.json
            if os.path.exists(os.path.join(entry_path, f"{entry}_meta.json")):
                direct_pmid_dirs.append((entry, entry_path))

        if direct_pmid_dirs:
            return direct_pmid_dirs

        # Case 2: paper_dir contains year folders, each containing PMID folders.
        for year in entries:
            year_dir = os.path.join(paper_dir, year)
            if not os.path.isdir(year_dir):
                continue

            year_entries = sorted(os.listdir(year_dir))
            for pmid in year_entries:
                pmid_dir = os.path.join(year_dir, pmid)
                if not os.path.isdir(pmid_dir):
                    continue
                if os.path.exists(os.path.join(pmid_dir, f"{pmid}_meta.json")):
                    pairs.append((str(pmid), pmid_dir))

        return pairs

    def _resolve_paper_files(self, pmid_dir: str, pmid: str) -> Dict[str, Optional[str]]:
        """Resolve supported file names for a PubMed paper directory."""
        candidates = {
            'meta_json': [f'{pmid}_meta.json'],
            'content_json': [f'{pmid}_content.json'],
            'content_md': [f'{pmid}_content.md'],
            'content_xml': [f'{pmid}_content.xml'],
        }
        resolved: Dict[str, Optional[str]] = {}
        for key, names in candidates.items():
            resolved[key] = None
            for name in names:
                path = os.path.join(pmid_dir, name)
                if os.path.exists(path):
                    resolved[key] = path
                    break
        return resolved

    def _extract_section_nodes(self, nodes: Any) -> List[Dict[str, Any]]:
        """Flatten nested section nodes into a list of title/content dicts."""
        sections: List[Dict[str, Any]] = []

        if not isinstance(nodes, list):
            return sections

        for node in nodes:
            if not isinstance(node, dict):
                continue

            title = str(node.get('title') or '').strip() or 'Untitled'
            content = self._content_to_text(node.get('content', []))
            if content:
                sections.append({'title': title, 'content': content})

            subsections = node.get('subsections', [])
            if isinstance(subsections, list) and subsections:
                sections.extend(self._extract_section_nodes(subsections))

        return sections

    def _read_pmids_from_file(self, file_path: str) -> List[str]:
        """Read PMIDs from a file (TXT or CSV)."""
        pmids = []

        try:
            if file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            value = str(row[0]).strip()
                            if value and value.lower() != "pmid":
                                pmids.append(value)
            else:
                # Assume TXT file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip comments
                            pmids.append(line)

        except Exception as e:
            raise ValueError(f"Error reading PMIDs file: {e}")

        return pmids

    def _load_paper_data(self, pmid_dir: str, pmid: str) -> Optional[Dict]:
        """Load and normalize paper data from metadata + parsed content files."""
        paths = self._resolve_paper_files(pmid_dir, pmid)
        meta_json = paths['meta_json']
        content_json = paths['content_json']
        content_md = paths['content_md']

        if not meta_json:
            return None

        try:
            with open(meta_json, 'r', encoding='utf-8') as meta_file:
                meta_dict = json.load(meta_file)

            parsed_dict: Dict[str, Any] = {}
            if content_json:
                with open(content_json, 'r', encoding='utf-8') as parsed_file:
                    parsed_dict = json.load(parsed_file)

            content_data = meta_dict.get('content', {})
            identity_data = meta_dict.get('identity', {})
            source_data = meta_dict.get('source', {})
            contributors_data = meta_dict.get('contributors', {})
            links_data = meta_dict.get('links', {})
            metadata_data = meta_dict.get('metadata', {})

            sections = self._extract_sections(parsed_dict)
            if not sections and isinstance(parsed_dict, dict) and parsed_dict.get('body'):
                sections = self._extract_section_nodes(parsed_dict.get('body', []))
            section_text = self._compose_section_text(sections, self._effective_sections())

            md_text = ""
            if os.path.exists(content_md):
                with open(content_md, 'r', encoding='utf-8') as content_file:
                    md_text = content_file.read().strip()

            text_candidates = []
            if self.config.text_source_priority == TextSourcePriority.PARSED_JSON_FIRST:
                text_candidates = [section_text, md_text]
            else:
                text_candidates = [md_text, section_text]

            selected_text = ""
            for candidate in text_candidates:
                if candidate and candidate.strip():
                    selected_text = candidate.strip()
                    break

            if not selected_text and content_data.get("abstract"):
                selected_text = str(content_data.get("abstract", "")).strip()

            full_text_content: Optional[str] = None
            if self.config.mode == MergeMode.METADATA_CONTENT:
                full_text_content = selected_text if selected_text else None

            has_full_text = bool((section_text and section_text.strip()) or (md_text and md_text.strip()))

            if not has_full_text:
                print(f"Warning: PMID {pmid} has no parsed text content; content will be None.")

            if sections and not content_data.get("abstract"):
                abstract = self._extract_abstract_from_sections(sections)
                if abstract:
                    content_data = dict(content_data)
                    content_data["abstract"] = abstract

            return {
                'pmid': pmid,
                'identity': identity_data,
                'content': content_data,
                'source': source_data,
                'contributors': contributors_data,
                'links': links_data,
                'metadata': metadata_data,
                'parsed': {
                    'title': parsed_dict.get('title') if isinstance(parsed_dict, dict) else None,
                    'body': parsed_dict.get('body', []) if isinstance(parsed_dict, dict) else [],
                },
                'text_sections': sections,
                'full_text': full_text_content,
                'has_full_text': has_full_text,
                'provenance': {
                    'pmid_dir': pmid_dir,
                    'meta_json_path': meta_json,
                    'content_json_path': content_json,
                    'content_md_path': content_md,
                    'text_source_priority': self.config.text_source_priority.value,
                    'output_profile': self.config.output_profile.value,
                },
            }

        except Exception as e:
            self.stats['failed'] += 1
            self.stats['failed_pmids'].append(pmid)
            print(f"Warning: Error loading PMID {pmid}: {e}")
            return None

    def _merge_papers(self, papers: List[Dict], output_path: str) -> Dict:
        """Merge papers into output file."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Choose output format
        if self.config.output_format == OutputFormat.MARKDOWN:
            self._write_markdown(papers, output_path)
        elif self.config.output_format == OutputFormat.JSONL:
            self._write_jsonl(papers, output_path)
        else:
            self._write_plain_text(papers, output_path)

        return self.stats

    def _resolve_merged_output_path(self, output_path: str, source_name: str, jsonl: bool = False) -> str:
        """Resolve the final merged output path.

        If the caller provides a directory or a path without a file suffix,
        create a timestamped filename using the input source name.
        """
        suffix = '.jsonl' if jsonl else '.json'
        if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
            output_dir = output_path if output_path else '.'
            return os.path.join(output_dir, f"{source_name}_{_timestamp()}{suffix}")
        return output_path

    def _write_paper_json(self, pmid_dir: str, paper: Dict[str, Any]) -> None:
        """Write canonical merged JSON next to a single paper's source files."""
        pmid = str(paper.get('pmid', '')).strip() or 'pubmed'
        output_path = os.path.join(pmid_dir, f'{pmid}.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as handle:
                json.dump(_safe_copy(paper), handle, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write {output_path}: {e}")

    def _write_markdown(self, papers: List[Dict], output_path: str):
        """Write papers in Markdown format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Literature Corpus\n\n")
            f.write(f"Total papers: {len(papers)}\n\n")

            for idx, paper in enumerate(papers, 1):
                try:
                    self._write_paper_markdown(f, paper, idx)
                    self.stats['successful'] += 1

                    if idx < len(papers):
                        f.write("\n---\n\n")

                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"Warning: Error writing paper {paper['pmid']}: {e}")

        print(f"Markdown output saved to: {output_path}")

    def _write_paper_markdown(self, f, paper: Dict, idx: int):
        """Write a single paper in Markdown format."""
        identity = paper['identity']
        content = paper['content']
        source = paper['source']
        full_text = paper['full_text']

        # Header
        f.write(f"## Paper {idx}\n\n")

        # Metadata
        f.write(f"**PMID:** {identity.get('pmid', 'N/A')}  \n")
        f.write(f"**DOI:** {identity.get('doi', 'N/A')}  \n")
        f.write(f"**Title:** {identity.get('title', 'N/A')}  \n")
        f.write(f"**Journal:** {self._scalar(source.get('journal_title', 'N/A'))}  \n")
        f.write(f"**Publication Date:** {source.get('pub_date', 'N/A')}  \n")

        # Keywords and MeSH
        keywords = content.get('keywords', [])
        mesh_terms = content.get('mesh_terms', [])

        if keywords:
            f.write(f"**Keywords:** {', '.join(keywords)}  \n")

        if mesh_terms:
            f.write(f"**MeSH Terms:** {', '.join(mesh_terms)}  \n")

        if self.config.output_profile == OutputProfile.LLM and self.config.include_links_in_llm:
            links = paper.get('links', {}) or {}
            refs = links.get('refs', []) if isinstance(links, dict) else []
            f.write(f"**Ref Count:** {len(refs) if isinstance(refs, list) else 0}  \n")

        f.write("\n")

        # Abstract
        f.write("### Abstract\n\n")
        abstract = content.get('abstract')
        if abstract:
            f.write(self._normalize_abstract(abstract))
        else:
            f.write("*No abstract available.*\n")

        f.write("\n")

        # Full text
        if self.config.mode == MergeMode.METADATA_CONTENT:
            f.write("### Full Text\n\n")

            if full_text and full_text.strip():
                f.write(full_text)
            else:
                f.write("*Full text not available.*\n")

            f.write("\n")

    def _write_jsonl(self, papers: List[Dict], output_path: str):
        """Write papers in JSONL format (one JSON per line)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                try:
                    json_line = self._prepare_json_line(paper)
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                    self.stats['successful'] += 1

                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"Warning: Error writing paper {paper['pmid']}: {e}")

        print(f"JSONL output saved to: {output_path}")

    def _prepare_json_line(self, paper: Dict) -> Dict:
        """Prepare a paper as a JSON line based on output profile."""
        if self.config.output_profile == OutputProfile.ANALYSIS:
            row = _safe_copy(paper)
            row['profile'] = self.config.output_profile.value
            if self.config.mode == MergeMode.METADATA_ONLY:
                row['full_text'] = None
            selected = self._select_metadata_fields(paper)
            row['metadata_selected'] = selected
            return row

        # LLM profile
        links = paper.get('links', {}) if isinstance(paper.get('links', {}), dict) else {}
        if not self.config.include_links_in_llm:
            links = {}

        return {
            'profile': self.config.output_profile.value,
            'pmid': paper.get('pmid', ''),
            'title': (paper.get('identity', {}) or {}).get('title', ''),
            'doi': (paper.get('identity', {}) or {}).get('doi', ''),
            'pub_date': (paper.get('source', {}) or {}).get('pub_date', ''),
            'keywords': (paper.get('content', {}) or {}).get('keywords', []),
            'mesh_terms': (paper.get('content', {}) or {}).get('mesh_terms', []),
            'llm_text': paper.get('full_text') if self.config.mode != MergeMode.METADATA_ONLY else (paper.get('content', {}) or {}).get('abstract', ''),
            'has_full_text': paper.get('has_full_text', False),
            'links': links,
        }

    def _write_plain_text(self, papers: List[Dict], output_path: str):
        """Write papers in plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, paper in enumerate(papers, 1):
                try:
                    self._write_paper_plain_text(f, paper, idx)
                    self.stats['successful'] += 1

                    if idx < len(papers):
                        f.write("\n" + ("=" * 80) + "\n\n")

                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"Warning: Error writing paper {paper['pmid']}: {e}")

        print(f"Plain text output saved to: {output_path}")

    def _write_paper_plain_text(self, f, paper: Dict, idx: int):
        """Write a single paper in plain text format."""
        identity = paper['identity']
        content = paper['content']
        source = paper['source']
        full_text = paper['full_text']

        f.write(f"[Document {idx}]\n")
        f.write(f"PMID: {identity.get('pmid', 'N/A')}\n")
        f.write(f"Title: {identity.get('title', 'N/A')}\n")
        f.write(f"Journal: {self._scalar(source.get('journal_title', 'N/A'))}\n")
        f.write(f"Publication Date: {source.get('pub_date', 'N/A')}\n")

        f.write("\n[Abstract]\n")
        abstract = content.get('abstract')
        if abstract:
            f.write(self._normalize_abstract(abstract))
        else:
            f.write("No Abstract Available.\n")

        f.write(f"\n[Keywords]: {content.get('keywords', [])}\n")
        f.write(f"[MeSH Terms]: {content.get('mesh_terms', [])}\n")

        if self.config.mode != MergeMode.METADATA_ONLY:
            f.write("\n[Full Text Content]\n")
            if full_text and full_text.strip():
                f.write(full_text)
            else:
                f.write("No Full Text Available (Metadata only).\n")

        f.write("\n")

    def _normalize_abstract(self, abstract: str) -> str:
        """Normalize abstract text for better AI parsing."""
        # Split by sentences and ensure proper formatting
        sentences = abstract.split('.')
        normalized = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                normalized.append(sentence + '.')

        return '\n'.join(normalized)

    def _extract_sections(self, parsed_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(parsed_dict, dict):
            return []
        body = parsed_dict.get('body', [])
        if not isinstance(body, list):
            return []

        return self._extract_section_nodes(body)

    def _content_to_text(self, content_items: Any) -> str:
        if isinstance(content_items, list):
            parts = [str(item).strip() for item in content_items if str(item).strip()]
            return '\n'.join(parts)
        if content_items is None:
            return ''
        return str(content_items).strip()

    def _effective_sections(self) -> Optional[List[str]]:
        if self.config.include_sections:
            return [_normalize_name(item) for item in self.config.include_sections if item and str(item).strip()]

        if self.config.output_profile == OutputProfile.LLM:
            return DEFAULT_LLM_SECTIONS

        return None

    def _compose_section_text(self, sections: List[Dict[str, Any]], include_sections: Optional[List[str]]) -> str:
        if not sections:
            return ''

        normalized_set = set(include_sections or [])
        lines: List[str] = []
        for section in sections:
            title = str(section.get('title', 'Untitled')).strip() or 'Untitled'
            content = str(section.get('content', '')).strip()
            if not content:
                continue

            if normalized_set and _normalize_name(title) not in normalized_set:
                continue

            lines.append(f"## {title}")
            lines.append('')
            lines.append(content)
            lines.append('')

        return '\n'.join(lines).strip()

    def _extract_abstract_from_sections(self, sections: List[Dict[str, Any]]) -> str:
        for section in sections:
            if _normalize_name(section.get('title', '')) == 'abstract':
                return str(section.get('content', '')).strip()
        return ''

    def _scalar(self, value: Any) -> str:
        if isinstance(value, list):
            return ', '.join(str(v) for v in value)
        return str(value)

    def _select_metadata_fields(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            'identity': paper.get('identity', {}),
            'content': paper.get('content', {}),
            'source': paper.get('source', {}),
            'contributors': paper.get('contributors', {}),
            'links': paper.get('links', {}),
            'metadata': paper.get('metadata', {}),
        }

        if self.config.include_metadata_fields:
            requested = [str(item).strip() for item in self.config.include_metadata_fields if str(item).strip()]
        elif self.config.output_profile == OutputProfile.ANALYSIS:
            requested = ['all']
        else:
            requested = ['identity', 'content.keywords', 'content.mesh_terms', 'source']

        if 'all' in requested:
            return payload

        selected: Dict[str, Any] = {}
        for field in requested:
            value = self._get_by_path(payload, field)
            if value is not None:
                self._set_by_path(selected, field, value)
        return selected

    def _get_by_path(self, data: Dict[str, Any], path: str) -> Any:
        current: Any = data
        for key in path.split('.'):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _set_by_path(self, data: Dict[str, Any], path: str, value: Any) -> None:
        keys = path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def print_statistics(self):
        """Print merge statistics."""
        print("\nMerge Statistics:")
        print(f"  Total processed: {self.stats['total_processed']}")
        print(f"  Successful: {self.stats['successful']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Skipped: {self.stats['skipped']}")

        if self.stats['failed_pmids']:
            print(f"\nFailed PMIDs: {', '.join(self.stats['failed_pmids'])}")


def parse_and_merge_pubmed_meta(
    paper_dir: str,
    output_txt: str,
    year_range: Optional[tuple] = None
) -> Dict:
    """
    Backward-compatible function: Merge only metadata.

    Args:
        paper_dir: Directory containing papers
        output_txt: Output text file path
        year_range: Deprecated parameter, kept for backward compatibility

    Returns:
        Statistics dictionary
    """
    config = MergeConfig(
        mode=MergeMode.METADATA_ONLY,
        output_format=OutputFormat.PLAIN_TEXT
    )

    merger = PaperMerger(config)
    stats = merger.merge_from_directory(paper_dir, output_txt)
    merger.print_statistics()

    return stats


def parse_and_merge_pubmed_meta_and_content(
    paper_dir: str,
    output_txt: str,
    year_range: Optional[tuple] = None
) -> Dict:
    """
    Backward-compatible function: Merge metadata and content.

    Args:
        paper_dir: Directory containing papers
        output_txt: Output text file path
        year_range: Deprecated parameter, kept for backward compatibility

    Returns: 
        Statistics dictionary
    """
    config = MergeConfig(
        mode=MergeMode.METADATA_CONTENT,
        output_format=OutputFormat.PLAIN_TEXT
    )

    merger = PaperMerger(config)
    stats = merger.merge_from_directory(paper_dir, output_txt)
    merger.print_statistics()

    return stats


# for CLI usage
@app.command("pubmed-merge")
def merge_cmd(
    paper_dir: str = typer.Argument(..., help="Directory containing paper data (pubmed/year/pmid/structure)."),
    output: str = typer.Argument(..., help="Output file path for merged data."),
    pmid_file: Optional[str] = typer.Option(None, "--pmid-file", "-p", help="File containing PMIDs to merge (one per line). If not specified, merge all papers in directory."),
    mode: str = typer.Option("full", "--mode", "-m", help="Merge mode: 'meta' (metadata only), 'full' (metadata + content)."),
    format: str = typer.Option("md", "--format", "-f", help="Output format: 'md' (Markdown), 'jsonl' (JSON Lines), 'txt' (plain text)."),
    profile: str = typer.Option("analysis", "--profile", help="Output profile: 'analysis' (full metadata for modules), 'llm' (LLM-focused compact output)."),
    include_sections: Optional[str] = typer.Option(None, "--include-sections", help="Comma-separated section names to include from parsed JSON, e.g. 'abstract,introduction,results'."),
    metadata_fields: Optional[str] = typer.Option(None, "--metadata-fields", help="Comma-separated metadata field paths, e.g. 'identity,content.keywords,links'. Use 'all' for full metadata."),
    include_links: Optional[bool] = typer.Option(None, "--include-links/--exclude-links", help="Whether to include links in llm profile output. Ignored in analysis profile."),
    text_source: str = typer.Option("parsed-json-first", "--text-source", help="Text source priority: 'parsed-json-first' or 'parsed-md-first'."),
):
    """
    Merge PubMed paper data into unified format for AI analysis.

    This command merges paper metadata and/or full text content into a single
    file optimized for AI processing and analysis.

    Examples:
      - Merge all papers with content to Markdown:
        paperflow merge ./papers_dir ./merged_papers.md --mode full --format md

      - Merge papers from a PMID list file:
        paperflow merge ./papers_dir ./selected_papers.md --pmid-file pmids.txt --mode full
    """
    # Map mode string to enum
    mode_map = {
        'meta': MergeMode.METADATA_ONLY,
                'full': MergeMode.METADATA_CONTENT,
    }

    if mode not in mode_map:
        typer.echo(f"Error: Invalid mode '{mode}'. Use: meta, full")
        raise typer.Exit(code=1)

    merge_mode = mode_map[mode]

    # Map format string to enum
    format_map = {
        'md': OutputFormat.MARKDOWN,
        'jsonl': OutputFormat.JSONL,
        'txt': OutputFormat.PLAIN_TEXT
    }

    if format not in format_map:
        typer.echo(f"Error: Invalid format '{format}'. Use: md, jsonl, txt")
        raise typer.Exit(code=1)

    output_format = format_map[format]

    # Map profile string to enum
    profile_map = {
        'analysis': OutputProfile.ANALYSIS,
        'llm': OutputProfile.LLM,
    }

    if profile not in profile_map:
        typer.echo(f"Error: Invalid profile '{profile}'. Use: analysis, llm")
        raise typer.Exit(code=1)

    output_profile = profile_map[profile]

    text_source_map = {
        'parsed-json-first': TextSourcePriority.PARSED_JSON_FIRST,
        'parsed-md-first': TextSourcePriority.PARSED_MD_FIRST,
    }

    if text_source not in text_source_map:
        typer.echo("Error: Invalid text-source. Use: parsed-json-first, parsed-md-first")
        raise typer.Exit(code=1)

    include_sections_list = [item.strip() for item in include_sections.split(',')] if include_sections else None
    metadata_fields_list = [item.strip() for item in metadata_fields.split(',')] if metadata_fields else None

    # Create merge config
    config = MergeConfig(
        mode=merge_mode,
        output_format=output_format,
        output_profile=output_profile,
        include_sections=include_sections_list,
        include_metadata_fields=metadata_fields_list,
        include_links_in_llm=(include_links if include_links is not None else False),
        text_source_priority=text_source_map[text_source],
    )

    # Create merger and execute
    merger = PaperMerger(config)

    typer.echo(f"Merging papers from: {paper_dir}")

    try:
        if pmid_file:
            # Merge from PMID list file
            stats = merger.merge_from_file_list(paper_dir, pmid_file, output)
        else:
            # Merge all papers in directory
            stats = merger.merge_from_directory(paper_dir, output)

        merger.print_statistics()

        if stats['successful'] > 0:
            typer.secho(f"Successfully merged {stats['successful']} papers.", fg=typer.colors.GREEN)

        if stats['failed'] > 0:
            typer.secho(f"{stats['failed']} papers failed to merge.", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.echo(f"Error during merge: {e}")
        raise typer.Exit(code=1)
