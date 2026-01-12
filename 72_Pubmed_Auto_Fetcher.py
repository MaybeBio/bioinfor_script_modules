import argparse
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from Bio import Entrez, Medline
import traceback
from dataclasses import dataclass, field, asdict


# All structure should be Raw Data -> Fetcher -> Paper Object -> JSON/Database -> AI Analyzer

@dataclass(frozen=True)
class PaperIdentity:
    """
    Description
    ----------
        Identity of the paper, storing basic identification information
    
    Args
    ----------
        pmid (str): pubmed ID of the paper
        doi (str): 默认为空字符串 digital object identifier of the paper, default is an empty string
        title (str): title of the paper, default is an empty string
    """
    pmid: str 
    doi: str = ""
    title: str = ""

@dataclass(frozen=True)
class PaperContent:
    """
    Description
    ----------
        Content of the paper, storing main textual information, important for NLP tasks like ontology construction.

    Args
    ----------
        abstract (str): abstract of the paper
        keywords (List[str]): list of keywords associated with the paper
        mesh_terms (List[str]): list of MeSH terms associated with the paper
        pub_types (List[str]): list of publication types of the paper, e.g., Journal Article, Review, etc.
    
    """
    abstract: str
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    pub_types: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PaperContributors:
    """
    Description
    ----------
        Contributors of the paper, storing information about authors and affiliations.
        Data is separated by source format (Medline vs XML) to preserve structure.

    Args
    ----------
        medline (Dict[str, List[str]]): Flat lists extracted from Medline format (or flattened from XML).
                                        Keys: 'full_names', 'short_names', 'auids', 'affiliations'.
        xml (List[Dict[str, Any]]): Structured author list with detailed fields (from XML).
    """
    medline: Dict[str, List[str]] = field(default_factory=dict)
    xml: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class PaperSource:
    """
    Description
    ----------
        Source information of the paper, storing journal and publication details.
    
    Args
    ----------
        journal_title (str): title of the journal
        journal_abbrev (str): abbreviation of the journal
        pub_date (str): publication date of the paper
        pub_year (str): publication year of the paper
        pub_types (List[str]): list of publication types of the paper, e.g., Journal Article, Review, etc. ⚠️ SAME as Pub_types in PaperContent

    
    TODOS
    ----------
    """
    # journal: Dict[str, str] = field(default_factory=dict)
    journal_title: str = ""
    journal_abbrev: str = ""
    pub_date: str = ""
    pub_year: str = ""
    pub_types: List[str] = field(default_factory=list)

@dataclass
class PaperLinks:
    """
    Description
    ----------
        Stores linked data from other NCBI databases (retrieved via ELink).
        This is crucial for the "Deep Ontology" platform.
    
    Args
    ----------
        cites (List[str]): List of PMIDs that cite this paper (pubmed_pubmed_citedin).
        refs (List[str]): List of PMIDs that this paper cites (pubmed_pubmed_refs).
        similar (List[str]): List of related articles (pubmed_pubmed).Similar PubMed articles, obtained by matching text and MeSH terms
        review (List[str]): List of review articles (pubmed_pubmed_reviews).
        entrez (Dict[str, List[str]]): Dictionary of other internal Entrez Cross-database links, keys are linkname, values are lists of linked UIDs.
        external (List[Dict[str, str]]): List of external database links
        fetch_timestamp (str): Timestamp when the linked data was fetched ⚠️.

        pubmed_pmc_local                         | Free full text articles in PMC
        pubmed_pmc                               | Free full text articles in PMC
    
    Notes
    ----------
    - 1, Since links may change over time, we store the fetch timestamp for reference.
    That means we need to update link data(e.g. cites) periodically for each paper in our database, and update the fetched article 
    """

    cites: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    similar: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)
    entrez: Dict[str, List[str]] = field(default_factory=dict)
    external: List[Dict[str, str]] = field(default_factory=list)
    full_text_pmc: List[str] = field(default_factory=list)
    fetch_timestamp: str = ""

@dataclass(frozen=True)
class PaperMetadata:
    """
    Description
    ----------
        Metadata of the paper, storing additional information about the fetching process.
    
    Args
    ----------
        entrez_date (str): Entrez date of the paper, used for incremental updates
        fetched_at (str): timestamp when the paper was fetched
        source_format (str): format of the source data, e.g., Medline, XML
    
    Notes
    ----------
    - 1, entrez_date is crucial for incremental updates, as it indicates when the paper was added to PubMed.
    - 2, ⚠️ All information in this class is related to the fetching process, not the paper content itself.
    """
    entrez_date: str = ""
    fetched_at: str = ""
    source_format: str = ""

@dataclass
class Paper_MetaData:
    identity: PaperIdentity
    content: PaperContent
    contributors: PaperContributors
    source: PaperSource
    metadata: PaperMetadata
    links: PaperLinks = field(default_factory=PaperLinks)

    def to_dict(self):
        return asdict(self)

@dataclass
class Paper_TextData:
    text: str

@dataclass
class Paper:
    Meta: Paper_MetaData
    Text: Paper_TextData

class PubmedFetcher:
    def __init__(self, output_dir: str, entrez_email: str,api_key: str, batch_size: int = 50):
        """
        Description
        -----------
        initializes the PubmedFetcher with output directory, Entrez email, and batch size.

        Args
        -----
        output_dir (str): Directory to save the fetched JSON files.
        entrez_email (str): Email address for NCBI Entrez. It is required by NCBI and should be set to a valid email.
        api_key (str): NCBI API Key for higher rate limits (10 req/sec).
        batch_size (int): Number of articles to fetch per batch, 50~100 is recommended.

        """
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.entrez_email = entrez_email
        self.api_key = api_key

        # Global setting for Entrez email (necessary for Biopython Entrez)
        Entrez.email = entrez_email

        if api_key:
            Entrez.api_key = api_key
            print(f"✅ NCBI API Key set successfully. Rate limit increased to 10 req/s.")
        
        # make sure that output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def query_search(self, query: str) -> Dict[str, Any]:
        """
        Description
        -----------
        Search for the first time in PubMed using the given query to get the count list of results and WebEnv info and QueryKey.

        Args
        -----
        query (str): The search query string.

        Returns
        -------
        Dict[str, Any]: A dictionary containing the count list of results, WebEnv, and QueryKey.

        Notes
        -----
        - 1, why ues retmax=0? Because we only want the count and WebEnv, not the actual IDs ——> we do not need to retrieve the actual IDs at this first stage.
        - 2, usehistory="y" is crucial as it tells the server to remember the search results for later retrieval using WebEnv.
        """
        print(f"Now searching PubMed with query [{query}] at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        try:
            # set Entrez.email before making requests
            # Entrez.email = self.entrez_email

            # usehistory="y" to enable WebEnv
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y")
            results = Entrez.read(handle)
            handle.close()
            
            count = int(results["Count"])
            print(f"found {count} related articles about [{query}] at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            
            return {
                "count": count,
                "webenv": results["WebEnv"],
                "query_key": results["QueryKey"]
            }
        except Exception as e:
            print(f"Search failed: {e}")
            return {"count": 0}

    def get_pubmedIDs_from_query(self, query_meta: str, retmax: int = 500) -> List[str]:
        """
        Description
        -----------
        Get all PubMed IDs from a query search using the history.

        Args
        -----
        query_meta (str): Metadata from the query search containing count, WebEnv, and QueryKey. Just the return 
        value of query_search function.
        retmax (int): Batch size for fetching PubMed IDs to retrieve. Default is 500.

        Returns
        -------
        List[str]: A list of PubMed IDs retrieved from the query.
        """

        count = query_meta.get("count", 0)
        webenv = query_meta.get("webenv", "")
        query_key = query_meta.get("query_key", "")

        if count == 0 or not webenv or not query_key:
            print(f"No PMIDs to fetch due to failure in step query_search at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            return []

        print(f"Retrieving {count} PMIDs from history server at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")

        all_pmids = []

        # Fetch in batches according to retmax
        for start in range(0, count, retmax):
            end = min(count, start + retmax)
            print(f"Fetching PMIDs {start + 1} to {end} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            
            try:
                handle = Entrez.esearch(
                    db="pubmed",
                    retstart=start,
                    retmax=retmax,
                    webenv=webenv,
                    query_key=query_key
                )
                results = Entrez.read(handle)
                handle.close()
                
                batch_pmids = results.get("IdList", [])
                all_pmids.extend(batch_pmids)
                
                print(f"  -> Retrieved {len(batch_pmids)} PMIDs in this batch.")
                
                # Polite delay to avoid overwhelming NCBI servers
                time.sleep(1)
                
            except Exception as e:
                print(f"Fetching PMIDs {start + 1} to {end} failed: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                # delay before retrying, and longer delay
                time.sleep(2)
                continue
        
        # Final count check
        print(f"Total PMIDs retrieved: {len(all_pmids)} out of {count} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        return all_pmids

    def fetch_from_query(self, query_meta: Dict[str, Any]) -> List[Paper]:
        """
        Description
        -----------
        Use the query search metadata to fetch articles in batches and save them as JSON files.

        Args
        -----
        query_meta (Dict[str, Any]): Metadata from the query search containing count, WebEnv, and QueryKey. Just the return \
        value of query_search function.

        Notes
        -----
        - 1, The best strategy to collect large number of articles:

        """
        count = query_meta.get("count", 0)
        webenv = query_meta.get("webenv", "")
        query_key = query_meta.get("query_key", "")
        
        if count == 0 or not webenv or not query_key:
            print(f"No articles to fetch due to failure in step query_search at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            return []

        # Fetch in batches according to batch_size
        for start in range(0, count, self.batch_size):
            end = min(count, start + self.batch_size)
            print(f"Fetching articles {start + 1} to {end} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            
            try:
                # Use efetch to get detailed information in Medline format
                # AS for rettype and retmode, please refer to: 
                # https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly
                fetch_handle_medline = Entrez.efetch(
                    db="pubmed",
                    rettype="medline",
                    retmode="text",
                    retstart=start,
                    retmax=self.batch_size,
                    webenv=webenv,
                    query_key=query_key
                )
                # and we also fetch xml format data
                fetch_handle_xml = Entrez.efetch(
                    db="pubmed",
                    retmode="xml",
                    retstart=start,
                    retmax=self.batch_size,
                    webenv=webenv,
                    query_key=query_key
                )

                # Parse Medline format data and xml format data
                data_medline = Medline.parse(fetch_handle_medline)
                # data_Medline now is a generator, convert to list will get all data
                # Now records_medline is a list of batch_size medline records
                records_medline = list(data_medline)

                # ⚠️ For xml, We read instead of parsing here, so we must be careful about the batch size and memory usage!
                data_xml = Entrez.read(fetch_handle_xml)
                # data_xml now is a huge dict, it only contains 2 keys:'PubmedBookArticle'(none) and 'PubmedArticle'(useful), we use latter
                articles_list = data_xml['PubmedArticle']

                fetch_handle_medline.close()
                fetch_handle_xml.close() 
                
                print(f"  -> Retrieved {len(records_medline)} Medline records and {len(articles_list)} Xml articles. Please check whether they equal and the efetch number here with esearch count.")
                
                # Now we start to parse each record, note that each record is a PubMed article in Medline format
                parsed_articles: List[Paper] = []

                # we parse medline and xml at the same time
                for record,article in zip(records_medline, articles_list):
                    # one single record
                    parsed_medline = self.parse_medline_record(record)
                    parsed_xml = self.parse_article_xml(article)

                    if parsed_medline:
                        # merge medline and xml parsed results
                        # ⚠️ Note that parsed_medline is the main body, we only update contributors from parsed_xml
                        parsed_medline.contributors.xml = parsed_xml
                        parsed_articles.append(parsed_medline)

                
                # --- Fetch Linked Data ---
                if parsed_articles:
                    current_batch_pmids = [p.identity.pmid for p in parsed_articles if p.identity.pmid] # list of pmids
                    # Batch fetch all links for the current batch of PMIDs
                    links_map = self.fetch_linked_data_for_batch_pmid(current_batch_pmids) # Dict[str, PaperLinks]
                    
                    # Map back to Paper objects
                    for p in parsed_articles:
                        if p.identity.pmid in links_map:
                            p.links = links_map[p.identity.pmid]
                
                # ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ 保存这一批次为单独的 JSON 文件
                if parsed_articles:
                    self.save_batch_to_json(parsed_articles, start)
                
                # Polite delay to avoid overwhelming NCBI servers
                time.sleep(1)
                
            except Exception as e:
                print(f"Fetching articles {start + 1} to {end} failed: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                continue

    def fetch_from_pmid_list(self, pmid_list: List[str]):
        """
        直接根据 PMID 列表下载文献详情。
        """
        count = len(pmid_list)
        if count == 0:
            print("PMID 列表为空。")
            return

        print(f"开始下载 {count} 篇指定 PMID 的文献...")

        # 按照 batch_size 循环获取
        for start in range(0, count, self.batch_size):
            end = min(count, start + self.batch_size)
            batch_pmids = pmid_list[start:end]
            print(f"正在获取第 {start + 1} 到 {end} 篇 (PMID: {batch_pmids[0]} ...)...")
            
            try:
                # 使用 efetch 直接通过 id 参数获取
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    rettype="xml",
                    retmode="xml",
                    id=",".join(batch_pmids)
                )
                data = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                # 解析这一批次的数据
                parsed_articles: List[Paper] = []
                if 'PubmedArticle' in data:
                    for article in data['PubmedArticle']:
                        parsed = self.parse_article_xml(article)
                        if parsed:
                            parsed_articles.append(parsed)
                
                # --- NEW: Fetch Linked Data ---
                if parsed_articles:
                    current_batch_pmids = [p.identity.pmid for p in parsed_articles if p.identity.pmid]
                    links_map = self.fetch_linked_data(current_batch_pmids)
                    for p in parsed_articles:
                        if p.identity.pmid in links_map:
                            p.links = links_map[p.identity.pmid]
                # ------------------------------

                # 保存这一批次为单独的 JSON 文件
                if parsed_articles:
                    self.save_batch_to_json(parsed_articles, start)
                
                # 礼貌性延时
                time.sleep(1)
                
            except Exception as e:
                print(f"获取批次 {start} 出错: {e}")
                continue

    def parse_medline_record(self, medline_record: Dict[str, Any]) -> Optional[Paper]:
        """
        Description
        -----------
        Parse one single Medline record into a structured dictionary to get relevant fields for construction of Ontology of this article.
        
        Args
        ----
        medline_record: Dict[str, Any]
            A single Medline record as parsed by Bio.Medline.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing structured information extracted from the Medline record.
        
        Notes
        -----
        - 1, For medline record field, refer to http://zatoka.icm.edu.pl/OVIDWEB/fldguide/medline.htm, https://medialab.github.io/sciencescape/medline_utils/
        - 2, the INPUT medline_record should be a single record of medline parsing result:
            handle = Entrez.efetch(db="pubmed",rettype="medline",retmode="text")
            records = Medline.parse(handle)
            medline_records = list(records)
            for medline_record in medline_records: where each medline_record is the input here.
        """

        try:
            # --- 1, Metadata extraction --- 
            pmid = medline_record.get("PMID", "")
            title = medline_record.get("TI", "")

            # ⚠️ get DOI from AID field if available (more reliable than LID)
            # AID and LID fields could be str or list
            doi = ""
            aids = medline_record.get("AID", [])
            if isinstance(aids, str):
                aids = [aids]
            for aid in aids:
                if "[doi]" in aid:
                    doi = aid.replace("[doi]", "").strip()
                    break
            
            # alternative DOI extraction from LID field if AID not available
            if not doi and 'LID' in medline_record:
                lids = medline_record.get("LID", [])
                if isinstance(lids, str):
                    lids = [lids]
                for lid in lids:
                    if '[doi]' in lid:
                        doi = lid.replace("[doi]", "").strip()
                        break

            # --- 2, Content extraction --- 
            abstract = medline_record.get("AB", "")

            # keywords: OT(Other Terms) field, always converted to list
            keywords = medline_record.get("OT", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            
            # MeSH terms: MH field, always converted to list
            # ⚠️ A Question: what's the difference between OT and MH fields?
            mesh_terms = medline_record.get("MH", [])
            if isinstance(mesh_terms, str):
                mesh_terms = [mesh_terms]

            # Publication Type: PT field, e.g., Journal Article, Review, etc.
            pub_types = medline_record.get("PT", [])
            if isinstance(pub_types, str):
                pub_types = [pub_types]

            # 3, ---- Entity: Authors & Affiliation extraction ---
            # Authors: FAU (Full Author Name) field
            full_author_names = medline_record.get("FAU", [])
            short_author_names = medline_record.get("AU", [])

            # Author Identifiers (e.g., ORCID): AUID field
            auids = medline_record.get("AUID", [])
            # Affiliations: AD field
            affiliations = medline_record.get("AD", [])

            # Ensure all author-related fields are lists
            for i in [full_author_names, short_author_names, auids, affiliations]:
                if isinstance(i, str):
                    i = [i]


            # Construct author entities/dictionaries
            # ⚠️ A Question: How to correctly map authors to their affiliations and identifiers?
            # we can not do this perfectly without more information
            # let's try xml format output

            # 4, ---- Entity: Journal & Publication extraction ---
            journal_title = medline_record.get("JT", ""), # Journal Title
            journal_abbrev = medline_record.get("TA", ""), # Journal Abbreviation

            # Date of Publication: DP field
            pub_date = medline_record.get("DP", "")
            pub_year = pub_date[:4] if len(pub_date) >= 4 else ""

            # Entrez Date: EDAT field, used for incremental updates (录入日期, 用于增量更新)
            entrez_date = medline_record.get("EDAT", "")
                 
            
            # ⚠️ Further fields can be extracted as needed

            return Paper(
                identity=PaperIdentity(pmid=pmid, doi=doi, title=title),
                content=PaperContent(abstract=abstract, 
                                     keywords=keywords, 
                                     mesh_terms=mesh_terms, 
                                     pub_types=pub_types),
                contributors=PaperContributors(
                    medline={
                        "full_names": full_author_names,
                        "short_names": short_author_names,
                        "auids": auids,
                        "affiliations": affiliations,
                    }
                ),
                source=PaperSource(
                    journal_title=journal_title,
                    journal_abbrev=journal_abbrev,
                    pub_date=pub_date,
                    pub_year=pub_year,
                    pub_types=pub_types,
                ),
                metadata=PaperMetadata(
                    entrez_date=entrez_date,
                    fetched_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    source_format="Medline"
                )
            )
        
        except Exception as e:
            print(f"Error parsing Medline record PMID {medline_record.get('PMID', 'Unknown')}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            traceback.print_exc()
            return None

    def parse_article_xml(self, article: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Description
        -----------
        Parse one single article in XML format into a structured dictionary to get relevant fields for construction of Ontology of this article.

        Args
        ----
        article: Any
            A single article as parsed by Entrez.read() from XML format.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing structured author information extracted from the article XML.
        
        Notes
        -----
        - 1, the INPUT article should be a single article of xml parsing result:
            handle = Entrez.efetch(db="pubmed",retmode="xml")
            data = Entrez.read(handle)
            articles_list = data['PubmedArticle']
            for article in articles_list: where each article is the input here.
        - 2, Note that we only extract author information here, other fields can be added as needed but medline format already covers most.
        """
        try:
            medline = article.get('MedlineCitation', {})
            article_data = medline.get('Article', {})
            pmid = str(medline.get('PMID', 'Unknown')) # defensive coding: in case to trace back errors

            # --- Authors & Affiliations ---
            authors_structured = []
            full_names_list = []
            short_names_list = []
            auids_list = []
            affiliations_list = []

            if 'AuthorList' in article_data:
                for author in article_data['AuthorList']:
                    # 1. Name Extraction
                    last_name = author.get('LastName', '')
                    fore_name = author.get('ForeName', '')
                    initials = author.get('Initials', '')
                    
                    # Construct Full Name (Medline FAU format: LastName, ForeName)
                    # else if only one of them exists, use that
                    full_name = f"{last_name}, {fore_name}" if last_name and fore_name else last_name or fore_name
                    if full_name: full_names_list.append(full_name)
                    
                    # Construct Short Name (Medline AU format: LastName Initials)
                    short_name = f"{last_name} {initials}" if last_name and initials else last_name or initials
                    if short_name: short_names_list.append(short_name)

                    # 2. Identifier Extraction (e.g. ORCID)
                    current_auids = []
                    if 'Identifier' in author:
                        # Note that author['Identifier'] could be a empty list or a list of identifiers
                        identifiers = author['Identifier']
                        # Defensive coding: handle both list and single item
                        if len(identifiers) > 0:
                            # it means that this author has at least one identifier
                            current_auids.append(str(identifiers[0]))

                    # 3. Affiliation Extraction
                    current_affiliations = []
                    if 'AffiliationInfo' in author:
                        aff_info = author['AffiliationInfo']
                        for aff in aff_info:
                            aff_text = aff.get('Affiliation', '')
                            if aff_text:
                                current_affiliations.append(aff_text)
                    
                    authors_structured.append({
                        "full_name": full_name,
                        "short_name": short_name,
                        "identifiers": current_auids,
                        "affiliations": current_affiliations,
                    })
            
            return authors_structured

        except Exception as e:
            print(f"Error parsing article XML PMID {pmid}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            traceback.print_exc()
            return None

    def fetch_linked_data_for_single_pmid(self, pmid: str) -> Dict[str, PaperLinks]:
        """
        Description
        -----------
        Uses ELink to fetch all connected data for a single PMID.
        
        Args
        ----
        pmid: str
            A single PubMed ID to fetch linked data for.

        Returns
        -------
        Dict[str, PaperLinks]
            A dictionary mapping the PMID to its corresponding PaperLinks object containing linked data.

        Notes
        -----
        - 1, ELink Strategy:
            1. 'acheck' to discover available internal Entrez links (Gene, Protein, etc.).
            2. 'neighbor' to fetch the actual IDs for those internal links. (1 and 2 are combined here)
            3. 'llinks' to fetch external URLs (LinkOuts) for datasets, full text, etc. (3 is separate)
        """
        if not pmid: 
            return {}
        
        # Map PMID -> PaperLinks
        links_map = {pmid: PaperLinks()}
        
        # --- Part A: Internal Entrez Links (Discovery & Fetching) ---
        # 1. Discovery (acheck)
        # Note: acheck returns LinkSetDbHistory
        try:
            handle = Entrez.elink(dbfrom="pubmed", id=pmid, cmd="acheck")
            acheck_results = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"Error in ELink acheck for {pmid}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            # Continue even if acheck fails
            acheck_results = [] 

        # Collect all unique LinkNames to query
        # tuple() like (db, linkname)
        links_to_fetch = set[Tuple[str, str]] = set()
        
        # ⚠️ The following links are always useful to fetch
        # cited_by /被他引
        links_to_fetch.add(("pubmed", "pubmed_pubmed_citedin"))
        # references /参考文献
        links_to_fetch.add(("pubmed", "pubmed_pubmed_refs"))
        # similar articles /相似文章
        links_to_fetch.add(("pubmed", "pubmed_pubmed"))
        # related reviews / 相关综述文章
        # ⚠️ But actually, We recommend you to unite all the articles about your PMID papers, and then efetch them using medline format, 
        # and finally filter them by PT field to get all reviews, which is more reliable.
        # So we do not guarantee that this link is complete about all reviews.
        links_to_fetch.add(("pubmed", "pubmed_pubmed_reviews"))

        if acheck_results:
            try:
                link_list = acheck_results[0]['IdCheckList']['IdLinkSet'][0]['LinkInfo']
                if link_list:
                    for link in link_list:
                        db = link.get('DbTo', '')
                        linkname = link.get('LinkName', '')

                        # Filter out useless UI links
                        # ⚠️ You may customize this filtering based on your needs
                        # After testing, we find that some links are not useful for ontology construction, such as:
                        # pubmed_pubmed_five (part of pubmed_pubmed), pubmed_pubmed_reviews_five (part of pubmed_pubmed_reviews)
                        # ExternalLink (link to external resources BUT returns nothing ?))
                        filter_list = ["pubmed_pubmed_five", "pubmed_pubmed_reviews_five", "ExternalLink"]
                        if linkname in filter_list:
                            continue 

                        if db and linkname:
                            links_to_fetch.add((db, linkname))
            except Exception as e:
                print(f"    Warning: Failed to parse acheck results for {pmid}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        
        print(f"  -> Deep mining {len(links_to_fetch)} types of internal connections for {pmid} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")

        # 2. Fetching (neighbor)
        for link in links_to_fetch:
            if link[0] == "LinkOut":
                # ("LinkOut", "ExternalLink") is handled again
                continue

            # for each link type, we initialize the db_link structure
            db_link = {"id": pmid, "db": link[0], "linkname": link[1], "links": []}
            try:
                print(f"     Fetching {link[1]} from {link[0]} for {pmid} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                handle = Entrez.elink(dbfrom="pubmed", id=pmid, db=link[0], linkname=link[1])
                results = Entrez.read(handle)
                handle.close()
                
                if not results: 
                    continue
                
                # Note that link[1] is the linkname, link[0] is the db
                # ⚠️ For different db and linkname, THE retured results structure may vary slightly

                # For db="pubmed" or "pmc", we can directly extract the linked PMIDs/PMCIDs, their uids are in results[0]['LinkSetDb'][0]['Link']
                for uid in results[0]['LinkSetDb'][0]['Link']:
                    db_link['links'].append(uid['Id'])

                # Now we have fetched all linked IDs for this link type
                if db_link['linkname'] == "pubmed_pubmed_citedin":
                    links_map[pmid].cites = db_link['links']
                elif db_link['linkname'] == "pubmed_pubmed_refs":
                    links_map[pmid].refs = db_link['links']
                elif db_link['linkname'] == "pubmed_pubmed":
                    links_map[pmid].similar = db_link['links']
                elif db_link['linkname'] == "pubmed_pubmed_reviews":
                    links_map[pmid].review = db_link['links']
                else:
                    # other internal links will be stored in entrez dict
                    links_map[pmid].entrez[db_link['linkname']] = db_link['links']

            except Exception as e:
                # since we force fetch some useful links above, some of them may not be available for certain PMIDs
                # so we may fail to fetch them at db_link['links'] update, just warn and continue
                print(f"    Warning: Failed to fetch {link[1]} from {link[0]} for {pmid}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                # If fetching fails in db_link['links'] update, we just continue, cause links_map[pmid].this_link is still an available empty list, we still can access it later
                continue

                
        # --- Part B: External LinkOuts (llinks) ---
        print(f"  -> Fetching external LinkOuts (Datasets, Full Text, etc.) for {pmid} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        try:
            handle = Entrez.elink(dbfrom="pubmed", id=pmid, cmd="llinks")
            llinks_results = Entrez.read(handle)
            handle.close()
            
            if llinks_results:
                urls_list = llinks_results[0]['IdUrlList']['IdUrlSet'][0]['ObjUrl']
                if urls_list:
                    links_map[pmid].external = urls_list
                
        except Exception as e:
            print(f"    Warning: Failed to fetch external LinkOuts for {pmid}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                
        return links_map

    def fetch_linked_data_for_batch_pmid(self, pmids: List[str]) -> Dict[str, PaperLinks]:
        """
        Description
        -----------
        Uses ELink to fetch all connected data for a batch of PMIDs.
        
        Args
        ----
        pmids: List[str]
            A list of PubMed IDs to fetch linked data for.

        Returns
        -------
        Dict[str, PaperLinks]
            A dictionary mapping each PMID to its corresponding PaperLinks object containing linked data.

        Notes
        -----
        - 1, This function internally calls fetch_linked_data_single for each PMID in the list.
        - 2, This batch processing helps to reduce the number of individual requests to NCBI servers.
        """

        if not pmids:
            return {}
        
        # Map PMID -> PaperLinks
        links_map = {pmid: PaperLinks() for pmid in pmids}

        # --- Part A: Internal Entrez Links (Discovery & Fetching) ---
        # 1. Discovery (acheck)
        # Note: acheck returns LinkSetDbHistory
        try:
            handle = Entrez.elink(dbfrom="pubmed", id=pmids, cmd="acheck")
            acheck_results = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"Error in batch Elink acheck for {len(pmids)} PMIDs: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            # Continue even if acheck fails
            acheck_results = []

        # Collect all unique LinkNames to query
        # tuple() like (db, linkname)
        links_to_fetch = set[Tuple[str, str]] = set()

        # ⚠️ The following links are always useful to fetch
        # cited_by /被他引
        links_to_fetch.add(("pubmed", "pubmed_pubmed_citedin"))
        # references /参考文献
        links_to_fetch.add(("pubmed", "pubmed_pubmed_refs"))
        # similar articles /相似文章
        links_to_fetch.add(("pubmed", "pubmed_pubmed"))
        # related reviews / 相关综述文章
        links_to_fetch.add(("pubmed", "pubmed_pubmed_reviews"))

        if acheck_results:
            for linkset in acheck_results:
                try:
                    link_list = linkset['IdCheckList']['IdLinkSet'][0]['LinkInfo']
                    if link_list:
                        for link in link_list:
                            db = link.get('DbTo', '')
                            linkname = link.get('LinkName', '')

                            # Filter out useless UI links
                            filter_list = ["pubmed_pubmed_five", "pubmed_pubmed_reviews_five", "ExternalLink"]
                            if linkname in filter_list:
                                continue 
                            if "combined" in linkname:
                                continue

                            if db and linkname:
                                links_to_fetch.add((db, linkname))
            
                except Exception as e:
                    print(f"    Warning: Failed to parse batch acheck results for {len(pmids)} PMIDs: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        
        print(f"  -> Deep mining {len(links_to_fetch)} types of internal connections for {len(pmids)} PMIDs at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")

        # 2. Fetching (neighbor)
        for link in links_to_fetch:
            if link[0] == "LinkOut":
                # ("LinkOut", "ExternalLink") is handled again
                continue

            # for each link type, we initialize the db_link structure
            try:
                print(f"     Fetching {link[1]} from {link[0]} for {len(pmids)} PMIDs at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                handle = Entrez.elink(dbfrom="pubmed", id=pmids, db=link[0], linkname=link[1])
                results = Entrez.read(handle)
                handle.close()
                
                if not results: 
                    continue

                # For each PMID in results
                for linkset in results:
                    source_id = "Unknown"
                    try:
                        source_id = str(linkset['IdList'][0]) # the source PMID
                        
                        if source_id not in links_map:
                            continue
                        
                        # for each link type, we initialize the db_link structure
                        db_link = {"id": source_id, "db": link[0], "linkname": link[1], "links": []}

                        # For db="pubmed" or "pmc", we can directly extract the linked PMIDs/PMCIDs, their uids are in linkset['LinkSetDb'][0]['Link']
                        for uid in linkset['LinkSetDb'][0]['Link']:
                            db_link["links"].append(uid['Id'])

                        # Now we have fetched all linked IDs for this link type, for this PMID
                        if db_link["linkname"] == "pubmed_pubmed_citedin":
                            links_map[source_id].cites = db_link['links']
                        elif db_link["linkname"] == "pubmed_pubmed_refs":
                            links_map[source_id].refs = db_link['links']
                        elif db_link["linkname"] == "pubmed_pubmed":
                            links_map[source_id].similar = db_link['links']
                        elif db_link["linkname"] == "pubmed_pubmed_reviews":
                            links_map[source_id].review = db_link['links']
                        else:
                            # other internal links will be stored in entrez dict
                            links_map[source_id].entrez[db_link['linkname']] = db_link['links']

                    except Exception as e:
                        print(f"    Warning: Failed to parse fetched links for {link[1]} from {link[0]} for {source_id}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                        continue


            except Exception as e:
                print(f"    Warning: Failed to fetch {link[1]} from {link[0]} for {len(pmids)} PMIDs: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                continue

        # --- Part B: External LinkOuts (llinks) ---
        print(f"  -> Fetching external LinkOuts (Datasets, Full Text, etc.) for {len(pmids)} PMIDs at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
        try:
            handle = Entrez.elink(dbfrom="pubmed", id=pmids, cmd="llinks")
            llinks_results = Entrez.read(handle)
            handle.close()
            
            if llinks_results:
                    for linkset in llinks_results:
                        source_id = "Unknown"
                        try:
                            source_id = linkset['IdUrlList']['IdUrlSet'][0]['Id'] # the source PMID
                            if source_id not in links_map:
                                continue
                            
                            urls_list = linkset['IdUrlList']['IdUrlSet'][0]['ObjUrl']
                            if urls_list:
                                links_map[source_id].external = urls_list

                        except Exception as e:
                            print(f"    Warning: Failed to parse external LinkOuts for {source_id}: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
                            continue
        except Exception as e:
            print(f"    Warning: Failed to fetch external LinkOuts for {len(pmids)} PMIDs: {e} at [{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")

        return links_map

    def fetch_pmc_full_text(self, pmc_ids: Union[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """
        Description
        -----------
        Fetch full text from PMC for given PMC IDs.
        Saves both raw XML and parsed text to the output directory.
        
        Args
        ----
        pmc_ids: Union[str, List[str]]
            A single PMC ID (e.g., "PMC8328303") or a list of PMC IDs.
            
        Returns
        -------
        Dict[str, Dict[str, str]]
            A dictionary where keys are PMC IDs and values are dictionaries containing:
            - 'xml_path': Path to the saved raw XML file.
            - 'text_path': Path to the saved parsed text file.
            - 'content': The parsed text content.
            - 'status': 'success' or 'error'.
            - 'error_message': Error message if failed.
        """
        if isinstance(pmc_ids, str):
            pmc_ids = [pmc_ids]
            
        results = {}
        
        # Create a sub-directory for PMC data
        pmc_dir = os.path.join(self.output_dir, "pmc_full_text")
        if not os.path.exists(pmc_dir):
            os.makedirs(pmc_dir)
            
        print(f"Fetching full text for {len(pmc_ids)} PMC articles...")
        
        for pmc_id in pmc_ids:
            # Clean PMC ID (remove 'PMC' prefix if present for filename, but Entrez usually handles both)
            clean_id = pmc_id.strip()
            # Entrez expects just the number or PMC+number. Let's keep as is but ensure clean filename
            safe_filename = clean_id.replace(":", "_").replace("/", "_")
            
            result_entry = {
                "xml_path": "",
                "text_path": "",
                "content": "",
                "status": "pending"
            }
            
            try:
                print(f"  -> Downloading {clean_id}...")
                # 1. Fetch Raw XML
                # db="pmc" is crucial
                handle = Entrez.efetch(db="pmc", id=clean_id, retmode="xml")
                # Read raw bytes for saving
                xml_content = handle.read()
                handle.close()
                
                # Save Raw XML
                xml_filename = f"{safe_filename}.xml"
                xml_path = os.path.join(pmc_dir, xml_filename)
                with open(xml_path, "wb") as f:
                    f.write(xml_content)
                result_entry["xml_path"] = xml_path
                
                # 2. Parse XML for Text
                # We need to re-parse the XML content using Entrez.read or similar to traverse it
                # Since we have bytes, we can use BytesIO
                from io import BytesIO
                handle = BytesIO(xml_content)
                try:
                    xml_records = Entrez.read(handle)
                except Exception as e:
                    # Fallback for some PMC XMLs that might fail strict parsing
                    print(f"     Warning: Entrez.read failed ({e}), trying to save only XML.")
                    result_entry["status"] = "xml_only"
                    result_entry["error_message"] = f"XML Parsing failed: {e}"
                    results[clean_id] = result_entry
                    continue

                if not xml_records:
                    result_entry["status"] = "empty"
                    results[clean_id] = result_entry
                    continue
                    
                article = xml_records[0]
                parsed_text = self._parse_pmc_xml_to_text(article)
                
                # Save Parsed Text
                txt_filename = f"{safe_filename}.txt"
                txt_path = os.path.join(pmc_dir, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(parsed_text)
                
                result_entry["text_path"] = txt_path
                result_entry["content"] = parsed_text
                result_entry["status"] = "success"
                
            except Exception as e:
                print(f"     Error fetching {clean_id}: {e}")
                result_entry["status"] = "error"
                result_entry["error_message"] = str(e)
            
            results[clean_id] = result_entry
            # Be polite
            time.sleep(0.5)
            
        return results

    def _parse_pmc_xml_to_text(self, article: Dict[str, Any]) -> str:
        """
        Helper function to recursively parse PMC XML structure into readable text.
        """
        output_lines = []
        
        # Title
        # The structure varies, sometimes it's in front -> article-meta -> title-group
        # But Entrez.read structure is specific.
        # Let's try to find body.
        
        if 'body' in article:
            body = article['body']
            self._recursive_section_parse(body, output_lines)
        else:
            output_lines.append("[No body content found in XML]")
            
        return "\\n".join(output_lines)

    def _recursive_section_parse(self, element: Any, output_lines: List[str], level: int = 1):
        """
        Recursive parser for PMC XML sections.
        """
        # Handle 'sec' (Section)
        # In Biopython Entrez.read, 'sec' is usually a list of dicts inside 'body' or other 'sec'
        
        # If we are passed the 'body' dict, it might have 'sec'
        if isinstance(element, dict) and 'sec' in element:
            sections = element['sec']
            # If it's a single dict (not list), wrap it
            if isinstance(sections, dict):
                sections = [sections]
                
            for sec in sections:
                # Title
                title = sec.get('title', '')
                if title:
                    # Clean title (it might be a StringElement with attributes)
                    title_text = str(title).strip()
                    output_lines.append(f"\\n{'#' * level} {title_text}")
                
                # Paragraphs 'p'
                if 'p' in sec:
                    paragraphs = sec['p']
                    if isinstance(paragraphs, dict) or isinstance(paragraphs, str):
                        paragraphs = [paragraphs]
                        
                    for p in paragraphs:
                        # p might contain mixed content. str(p) usually gets the text content in Biopython
                        text = str(p).replace('\\n', ' ').strip()
                        if text:
                            output_lines.append(f"{text}")
                
                # Recursive call for subsections
                self._recursive_section_parse(sec, output_lines, level + 1)

    def save_batch_to_json(self, articles: List[Paper], start_index: int):
        """
        将一批文章保存为 JSON 文件。
        文件名格式: batch_{timestamp}_{start_index}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{timestamp}_{start_index}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([article.to_dict() for article in articles], f, ensure_ascii=False, indent=2)
        
        print(f"  -> 已保存 {len(articles)} 篇文献到 {filename}")

def main():
    parser = argparse.ArgumentParser(description="自动化 PubMed 文献获取工具 (JSON版)")
    
    # 创建互斥组：要么搜索，要么从文件读取
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-q", "--query", type=str, help="搜索关键词")
    group.add_argument("-f", "--file", type=str, help="包含 PMID 列表的文件路径 (每行一个 PMID)")

    parser.add_argument("-o", "--outdir", type=str, default="pubmed_data_raw", help="输出目录")
    parser.add_argument("-n", "--batch_size", type=int, default=10, help="每个JSON文件包含的文献数量")
    parser.add_argument("--days", type=int, help="[可选] 仅获取最近 N 天录入的文献 (增量更新模式)")
    # 建议将此配置放入环境变量或配置文件中
    parser.add_argument("--email", type=str, default="luxunisgod123@gmail.com", help="Entrez Email")

    args = parser.parse_args()
    
    fetcher = PubmedFetcher(output_dir=args.outdir, entrez_email=args.email, batch_size=args.batch_size)

    if args.file:
        # 模式 A: 从文件读取 PMID 列表下载
        if not os.path.exists(args.file):
            print(f"错误: 文件 {args.file} 不存在。")
            return
        
        with open(args.file, 'r') as f:
            # 读取非空行，去除空白字符
            pmid_list = [line.strip() for line in f if line.strip()]
        
        fetcher.fetch_from_pmid_list(pmid_list)

    elif args.query:
        # 模式 B: 搜索并下载
        final_query = args.query
        if args.days:
            final_query = f"{args.query} AND \"last {args.days} days\"[Date - Entrez]"
            print(f"启用增量模式，查询修正为: {final_query}")
        
        # 1. 搜索
        search_meta = fetcher.query_search(final_query)
        
        # 2. 下载并保存
        fetcher.fetch_from_query(search_meta)

if __name__ == "__main__":
    main()
