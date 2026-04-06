# 给出1个pdb id，解析其对应的uniprot 序列信息，也就是将uniprot残基坐标、信息注释到pdb上

# 1.
# 参考: https://github.com/Grabyy/Sifts_Parser/tree/main
# 用法: curl --silent https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/7w1m.xml.gz | gunzip | python /data2/sifts_parse.py  
# 在当前目录下新建文件夹/data2/tsv_bank，给出每一个蛋白质entity的坐标映射tsv文件

from contextlib import closing
import csv
import gzip
import os

# import shutil
import sys
from urllib import request
from urllib.error import URLError
from xml.etree import cElementTree as ET

OUT_PATH = "./tsv_bank"
AA = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "HSE": "H",
    "HSD": "H",
    "HSP": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "PAQ": "Y",
    "AGM": "R",
    "PR3": "C",
    "DOH": "D",
    "CCS": "C",
    "GSC": "G",
    "GHG": "Q",
    "OAS": "S",
    "MIS": "S",
    "SIN": "D",
    "TPL": "W",
    "SAC": "S",
    "4HT": "W",
    "FGP": "C",
    "HSO": "H",
    "LYZ": "K",
    "FGL": "S",
    "PRS": "P",
    "DCY": "C",
    "LYM": "K",
    "GPL": "K",
    "PYX": "C",
    "PCC": "P",
    "EHP": "F",
    "CHG": "A",
    "TPO": "T",
    "DAS": "D",
    "AYA": "A",
    "TYN": "Y",
    "SVA": "S",
    "SCY": "C",
    "BNN": "A",
    "5HP": "E",
    "HAR": "R",
    "IAS": "D",
    "SNC": "C",
    "AHB": "N",
    "PTR": "Y",
    "PHI": "F",
    "NPH": "C",
    "PHL": "F",
    "SNN": "D",
    "A66": "A",
    "TYB": "Y",
    "PHD": "D",
    "MAA": "A",
    "APN": "A",
    "TYY": "Y",
    "TYT": "Y",
    "TIH": "A",
    "TRG": "K",
    "CXM": "M",
    "DIV": "V",
    "TYS": "Y",
    "DTH": "T",
    "MLE": "L",
    "CME": "C",
    "SHR": "K",
    "OCY": "C",
    "DTY": "Y",
    "2AS": "D",
    "AEI": "T",
    "DTR": "W",
    "OCS": "C",
    "CMT": "C",
    "BET": "G",
    "NLP": "L",
    "LLY": "K",
    "SCH": "C",
    "CEA": "C",
    "LLP": "K",
    "TRF": "W",
    "HMR": "R",
    "TYI": "Y",
    "TRO": "W",
    "NLE": "L",
    "BMT": "T",
    "BUC": "C",
    "PEC": "C",
    "BUG": "L",
    "SCS": "C",
    "NLN": "L",
    "MHO": "M",
    "CSO": "C",
    "FTR": "W",
    "DLE": "L",
    "TRN": "W",
    "CSE": "C",
    "CSD": "A",
    "OMT": "M",
    "CSA": "C",
    "DSP": "D",
    "CSB": "C",
    "DSN": "S",
    "SHC": "C",
    "CSX": "C",
    "YCM": "C",
    "CSZ": "C",
    "TRQ": "W",
    "CSW": "C",
    "EFC": "C",
    "CSP": "C",
    "CSS": "C",
    "CSR": "C",
    "CZZ": "C",
    "MSO": "M",
    "BTR": "W",
    "HLU": "L",
    "MGN": "Q",
    "HTI": "C",
    "TYQ": "Y",
    "4IN": "W",
    "M3L": "K",
    "C5C": "C",
    "HTR": "W",
    "MPQ": "G",
    "KCX": "K",
    "GLH": "E",
    "DIL": "I",
    "ACA": "A",
    "NEM": "H",
    "5CS": "C",
    "LYX": "K",
    "DVA": "V",
    "ACL": "R",
    "GLX": "Z",
    "MLZ": "K",
    "GLZ": "G",
    "SME": "M",
    "SMC": "C",
    "DLY": "K",
    "NEP": "H",
    "BCS": "C",
    "ASQ": "D",
    "SET": "S",
    "SEP": "S",
    "ASX": "B",
    "DGN": "Q",
    "DGL": "E",
    "MHS": "H",
    "SEG": "A",
    "ASB": "D",
    "ASA": "D",
    "SEC": "C",
    "SEB": "S",
    "ASK": "D",
    "GGL": "E",
    "ASI": "N",
    "SEL": "S",
    "CGU": "E",
    "C6C": "C",
    "ALO": "T",
    "ASL": "D",
    "LTR": "W",
    "CLD": "S",
    "CLE": "L",
    "GMA": "E",
    "1LU": "L",
    "CLB": "S",
    "MVA": "V",
    "S1H": "S",
    "DNP": "A",
    "SAR": "G",
    "FME": "M",
    "ALM": "A",
    "LEF": "L",
    "MEN": "N",
    "TPQ": "Y",
    "NMC": "G",
    "SBD": "S",
    "ALY": "K",
    "MME": "M",
    "GL3": "G",
    "ALS": "C",
    "SBL": "S",
    "2MR": "R",
    "CAY": "C",
    "3AH": "H",
    "DPR": "P",
    "CAS": "C",
    "NC1": "S",
    "HYP": "P",
    "FLA": "A",
    "LCX": "K",
    "MSE": "M",
    "IYR": "Y",
    "DPN": "F",
    "BAL": "A",
    "CAF": "C",
    "MSA": "G",
    "AIB": "A",
    "HIP": "H",
    "CYQ": "C",
    "PCA": "E",
    "DAL": "A",
    "BFD": "D",
    "DAH": "F",
    "HIC": "H",
    "CYG": "C",
    "DAR": "R",
    "CYD": "C",
    "IIL": "I",
    "CYM": "C",
    "CYL": "C",
    "CY3": "C",
    "CY1": "C",
    "HAC": "A",
    "143": "C",
    "DHI": "H",
    "CY4": "C",
    "YOF": "Y",
    "HPQ": "F",
    "SOC": "C",
    "DHA": "A",
    "2LU": "L",
    "MLY": "K",
    "TRW": "W",
    "STY": "Y",
    "MCL": "K",
    "BHD": "D",
    "NRQ": "Y",
    "ARM": "R",
    "PRR": "A",
    "ARO": "R",
}


class Sifts:
    """Ersatz de classe de parsing de fichier xml"""

    def __init__(self, sifts_id=None, stdin=None):
        """"""
        self.__filename__ = ""
        self.__protein__ = {}
        if stdin is not None:
            self.__load(stdin)
        elif sifts_id is not None:
            self.get_xml(sifts_id)

    def __set_filename(self, filename):
        """"""
        self.__filename__ = filename

    def get_filename(self):
        """"""
        return self.__filename__

    def __set_protein(self, protein):
        """"""
        self.__protein__ = protein

    def get_protein(self):
        """"""
        return self.__protein__

    def get_chains(self):
        """"""
        return [chain for chain in self.get_protein().keys()]

    def get_segment(self):
        """"""
        return [
            segment
            for chain in self.get_chains()
            for segment in self.get_protein()[chain].keys()
        ]

    def get_info(self):
        """"""
        return sorted(
            {
                elem
                for chain in self.get_chains()
                for segment in self.get_protein()[chain].keys()
                for elem in self.get_protein()[chain][segment].keys()
            }
        )

    def print_structure(self):
        """"""
        print(f"\n    Name : {self.get_filename()}")
        print(f"    Chain : {self.get_chains()}")
        print(f"    Segment : {self.get_segment()}")
        print(f"    Info : {self.get_info()}\n")

    def write_error(self, error, info, func):
        """"""
        if not os.path.exists("Error"):
            os.makedirs("Error")

        with open(f"Error/{func}.txt", "a+", encoding="utf-8") as f_out:
            f_out.write(f"Error : {error} ; Where : {info}\n")

    def code_3to1(self, code, seg_id):
        """Transforme le code d'un résidue de 3 lettre en 1 lettre"""
        try:
            return AA[code]
        except KeyError as error:
            self.write_error(error, seg_id, "code_3to1")
            return "X"

    def get_xml(self, pdb_id):
        """Define the URL to download the PDB assembly file"""
        url = f"ftp://anonymous:@ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz"
        print(f"Searching file at {url}")
        try:
            with closing(request.urlopen(url)) as gz_file:
                with gzip.open(gz_file, "rb") as xml_file:
                    self.__load(xml_file.read())
        except URLError as error:
            self.write_error(error, url, "get_xml")

    def __load(self, xml_file):
        """Fouille dans le xml et sort les information voulue"""
        root = ET.fromstring(xml_file)
        self.__set_filename(root.attrib["dbAccessionId"])
        protein = {}
        for entity in root.findall(
            "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}entity"
        ):
            chain_dic = {}
            chain_id = entity.attrib["entityId"]
            for segment in entity.findall(
                "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}segment"
            ):
                seg_dic = {}
                seg_id = segment.attrib["segId"]
                seg_length = 0
                db_source = []
                for l_res in segment.findall(
                    "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}listResidue"
                ):
                    for residue in l_res.findall(
                        "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}residue"
                    ):
                        for res in residue.findall(
                            "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}crossRefDb"
                        ):
                            if res.attrib["dbSource"] not in db_source:
                                db_source.append(res.attrib["dbSource"])
                                seg_dic[res.attrib["dbSource"] + "_seq"] = [
                                    "null"
                                ] * seg_length
                                seg_dic[res.attrib["dbSource"] + "_num"] = [
                                    "null"
                                ] * seg_length
                                seg_dic[res.attrib["dbSource"] + "_id"] = [
                                    res.attrib["dbAccessionId"]
                                ] * seg_length

                            if (
                                len(seg_dic[res.attrib["dbSource"] + "_seq"])
                                == seg_length
                            ):
                                seg_dic[res.attrib["dbSource"] + "_seq"] += [
                                    (
                                        lambda: res.attrib["dbResName"]
                                        if len(res.attrib["dbResName"]) == 1
                                        else self.code_3to1(
                                            res.attrib["dbResName"], seg_id
                                        )
                                    )()
                                ]
                                seg_dic[res.attrib["dbSource"] + "_num"] += [
                                    res.attrib["dbResNum"]
                                ]
                                seg_dic[res.attrib["dbSource"] + "_id"] += [
                                    res.attrib["dbAccessionId"]
                                ]
                        seg_length += 1

                chain_dic[seg_id] = seg_dic
                protein[chain_id] = chain_dic

        self.__set_protein(protein)
        print(f"{self.get_filename()} loaded......")

    def to_tsv(self, section=None, dir_path=os.getcwd()):
        """Ecrit les résultats sous forme d'un fichier tsv"""

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        section = section if section is not None else self.get_info()

        for chain in self.get_protein().values():
            for segment_id, segment in chain.items():
                with open(
                    f"{dir_path}/{segment_id}.tsv", "w", encoding="utf_8"
                ) as csvfile:
                    header = []
                    content = []
                    for elem_id, elem in segment.items():
                        if elem_id in section:
                            header.append(elem_id)
                            content.append(elem)

                    writer = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
                    writer.writerow(header)
                    writer.writerows(zip(*content))
                    print(f"{segment_id}.tsv created ..........")


def print_usage():
    """"""
    if sys.stdin.isatty():
        print("*****************************************************************")
        print("**                                                             **")
        print("** Usage : curl http:/myProtein.xml | gunzip | sifts_parser.py **")
        print("**                                                             **")
        print("*****************************************************************")
        sys.exit()


if __name__ == "__main__":
    print_usage()

    xml = Sifts(stdin=sys.stdin.read())
    xml.to_tsv(dir_path=OUT_PATH)


######################################################################################################

# 2.
# 
