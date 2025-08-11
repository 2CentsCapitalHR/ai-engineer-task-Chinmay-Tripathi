"""
Document Parser for Phase 2 – Fixed Version
Enhanced extraction, structure, and analysis for ADGM Corporate Agent
"""

from docx import Document
import re
import os
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.docx']
        self.nlp = None
        self._load_nlp_model()

        self.section_patterns = {
            'article': r'article\s+(\d+)\.?\s*(.+)',
            'clause': r'clause\s+(\d+)\.?\s*(.+)',
            'section': r'section\s+(\d+)\.?\s*(.+)',
            'part': r'part\s+([ivxlcdm]+|\d+)\.?\s*(.+)',
            'chapter': r'chapter\s+(\d+)\.?\s*(.+)',
            'whereas': r'whereas\s*[,;:]?\s*(.+)',
            'resolved': r'(?:resolved|it is resolved)\s*[,;:]?\s*(.+)',
            'definitions': r'definitions?\s*[,;:]?\s*(.+)?'
        }

        self.type_indicators = {
            'articles_of_association': {
                'primary': ['articles of association', 'company governance', 'board of directors'],
                'secondary': ['quorum', 'shareholders meeting', 'voting rights', 'share capital'],
                'patterns': [r'article\s+\d+', r'board\s+of\s+directors', r'general\s+meeting']
            },
            'memorandum_of_association': {
                'primary': ['memorandum of association', 'company objects', 'liability limited'],
                'secondary': ['registered office', 'authorized capital', 'company purposes'],
                'patterns': [r'objects?\s+of\s+(?:the\s+)?company', r'liability.*limited', r'registered\s+office']
            },
            'incorporation_application': {
                'primary': ['incorporation application', 'company registration', 'registration authority'],
                'secondary': ['proposed company name', 'registered address', 'initial directors'],
                'patterns': [r'application\s+for\s+incorporation', r'proposed\s+name', r'registration\s+authority']
            },
            'board_resolution': {
                'primary': ['board resolution', 'directors resolution', 'board meeting'],
                'secondary': ['resolved that', 'unanimously resolved', 'board unanimously'],
                'patterns': [r'resolved\s+that', r'board\s+resolution', r'directors?\s+resolve']
            },
            'register_members': {
                'primary': ['register of members', 'membership register', 'shareholder register'],
                'secondary': ['member details', 'share certificates', 'shareholding'],
                'patterns': [r'register\s+of\s+members', r'member\s+name', r'shares?\s+held']
            },
            'employment_contract': {
                'primary': ['employment contract', 'employment agreement', 'service agreement'],
                'secondary': ['salary', 'working hours', 'termination', 'probation'],
                'patterns': [r'employment\s+(?:contract|agreement)', r'salary.*payable', r'working\s+hours']
            },
            'commercial_agreement': {
                'primary': ['commercial agreement', 'business agreement', 'service agreement'],
                'secondary': ['terms and conditions', 'payment terms', 'delivery'],
                'patterns': [r'terms?\s+and\s+conditions', r'payment\s+terms', r'force\s+majeure']
            }
        }

    def _load_nlp_model(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except (ImportError, OSError):
            logger.warning("spaCy not available — using basic phrase extraction")
            self.nlp = None

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        try:
            doc = Document(file_path)
            filename = os.path.basename(file_path)

            paragraphs = self._extract_paragraphs(doc)
            tables = self._extract_tables(doc)
            sections = self._identify_sections(paragraphs)
            document_structure = self._analyze_document_structure(paragraphs, sections)

            full_text = ' '.join([p['text'] for p in paragraphs])
            section_texts = [
                sec['title'] + " " + " ".join(p['text'] for p in sec['paragraphs'])
                for sec in sections
            ]
            stats = {
                'word_count': len(full_text.split()),
                'paragraph_count': len(paragraphs),
                'section_count': len(sections),
                'table_count': len(tables),
                'sentence_count': self._count_sentences(full_text),
                'avg_words_per_paragraph': self._calculate_avg_words_per_paragraph(paragraphs),
                'avg_sentences_per_paragraph': self._calculate_avg_sentences_per_paragraph(paragraphs)
            }
            try:
                file_size = os.path.getsize(file_path)
            except Exception:
                file_size = 0

            return {
                'metadata': {
                    'filename': filename,
                    'file_size': file_size,
                    'parsing_timestamp': self._get_timestamp(),
                    'parser_version': 'enhanced_v2'
                },
                'structure': {
                    'paragraphs': paragraphs,
                    'tables': tables,
                    'sections': sections,
                    'section_texts': section_texts,
                    'document_structure': document_structure
                },
                'statistics': stats,
                'analysis': {
                    'text_quality': self._analyze_text_quality(paragraphs),
                    'legal_elements': self._identify_legal_elements(full_text),
                    'readability': self._calculate_readability(full_text),
                    'document_complexity': self._assess_complexity(paragraphs, sections)
                },
                'content': {
                    'full_text': full_text,
                    'key_phrases': self._extract_key_phrases(full_text),
                    'named_entities': self._extract_named_entities(full_text)
                }
            }
        except Exception as e:
            raise Exception(f"Failed to parse document: {str(e)}")

    def _extract_paragraphs(self, doc: Document) -> List[Dict[str, Any]]:
        out = []
        for i, p in enumerate(doc.paragraphs):
            text = p.text.strip()
            if text:
                out.append({
                    'index': i,
                    'text': text,
                    'word_count': len(text.split()),
                    'sentence_count': self._count_sentences(text),
                    'is_header': self._is_likely_header(text),
                    'is_list_item': self._is_list_item(text)
                })
        return out

    def _extract_tables(self, doc: Document) -> List[Dict[str, Any]]:
        tables_data = []
        for idx, table in enumerate(doc.tables):
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            tables_data.append({
                'index': idx,
                'rows': rows,
                'row_count': len(table.rows),
                'column_count': len(table.columns)
            })
        return tables_data

    def _identify_sections(self, paragraphs: List[Dict]) -> List[Dict[str, Any]]:
        sections, current = [], None
        for para in paragraphs:
            if para['is_header'] or self._matches_section_pattern(para['text']):
                if current: sections.append(current)
                current = {'title': para['text'], 'paragraphs': []}
            if current: current['paragraphs'].append(para)
        if current: sections.append(current)
        return sections

    def _analyze_document_structure(self, paragraphs, sections):
        return {
            'has_title': self._has_document_title(paragraphs),
            'has_introduction': any('introduction' in s['title'].lower() for s in sections),
            'has_definitions': any('definition' in s['title'].lower() for s in sections),
            'has_conclusion': bool(sections and any(w in sections[-1]['title'].lower() for w in ['conclusion', 'final'])),
            'section_hierarchy': {'total_levels': 1, 'max_level': 1},
            'structural_completeness': 1.0 if len(sections) >= 3 else 0.5,
            'organization_score': min(1.0, sum(1 for p in paragraphs if p['is_header']) / max(1, len(paragraphs)) * 5)
        }

    def _analyze_text_quality(self, paragraphs):
        all_text = ' '.join([p['text'] for p in paragraphs])
        return {
            'clarity_score': 0.8,
            'completeness_indicators': [],
            'formatting_consistency': {'formatting_score': 0.8}
        }

    def _identify_legal_elements(self, text):
        return {
            'legal_patterns': {'obligations': {'count': text.lower().count('shall')}} ,
            'compliance_indicators': [],
            'contract_elements': {'parties': 'party' in text.lower(), 'terms': 'terms' in text.lower()},
            'risk_indicators': []
        }

    def _calculate_readability(self, text):
        wc = len(text.split())
        sc = self._count_sentences(text)
        return {'complexity_score': round((wc / max(1, sc)) * 2, 2)}

    def _assess_complexity(self, paragraphs, sections):
        total_words = sum(p['word_count'] for p in paragraphs)
        return {'overall_complexity': 'high' if total_words > 2000 else 'medium'}

    def _count_sentences(self, text): return len(re.findall(r'[.!?]+', text))
    def _calculate_avg_words_per_paragraph(self, pars): return round(sum(p['word_count'] for p in pars) / max(1, len(pars)), 2)
    def _calculate_avg_sentences_per_paragraph(self, pars): return round(sum(p['sentence_count'] for p in pars) / max(1, len(pars)), 2)
    def _is_likely_header(self, text): return text.isupper() or bool(re.match(r'^(ARTICLE|SECTION|CLAUSE)\s+\d+', text))
    def _matches_section_pattern(self, text): return any(re.match(p, text, re.IGNORECASE) for p in self.section_patterns.values())
    def _is_list_item(self, text): return bool(re.match(r'^\d+\.|\([a-z]\)|[•\-\*]', text.strip()))
    def _get_timestamp(self): return datetime.now().isoformat()
    def _has_document_title(self, paragraphs): return bool(paragraphs and paragraphs[0]['is_header'])
    def _extract_key_phrases(self, text):
        if self.nlp: return [c.text for c in self.nlp(text).noun_chunks][:10]
        return list(set(re.findall(r'[A-Z][a-z]+(?: [A-Z][a-z]+)*', text)))[:10]
    def _extract_named_entities(self, text):
        if self.nlp: return [{'text': e.text, 'label': e.label_} for e in self.nlp(text).ents]
        return [{'text': t, 'label': 'UNKNOWN'} for t in list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)))]


def parse_document(file_path: str): 
    return DocumentParser().parse_document(file_path)

