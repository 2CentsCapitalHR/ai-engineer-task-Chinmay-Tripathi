import re
import pickle
import os
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docx import Document as DocxDocument
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_COLOR_INDEX
import tempfile, shutil

logger = logging.getLogger(__name__)

class DocumentClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.classifiers = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'svm': SVC(kernel='linear', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        self.is_trained = False
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.document_types = [
            'articles_of_association',
            'memorandum_of_association',
            'incorporation_application',
            'board_resolution',
            'register_members_directors',
            'employment_contract',
            'commercial_agreement',
            'licensing_application',
            'regulatory_filing',
            'compliance_policy',
        ]
        self.adgm_patterns = {
            'articles_of_association': {
                'primary_keywords': [
                    'articles of association', 'company governance', 'board of directors',
                    'shareholders', 'general meeting', 'quorum', 'voting rights'
                ],
                'secondary_keywords': [
                    'share capital', 'dividend', 'board meeting', 'annual general meeting',
                    'extraordinary resolution', 'ordinary resolution', 'director appointment'
                ],
                'legal_patterns': [
                    r'article\s+\d+',
                    r'board\s+of\s+directors',
                    r'general\s+meeting',
                    r'share\s+capital',
                    r'voting\s+rights'
                ],
                'section_indicators': ['governance', 'meetings', 'directors', 'shares']
            },
            'memorandum_of_association': {
                'primary_keywords': [
                    'memorandum of association', 'company objects', 'liability limited',
                    'registered office', 'authorized capital'
                ],
                'secondary_keywords': [
                    'company purposes', 'business activities', 'share classes',
                    'incorporation', 'company formation'
                ],
                'legal_patterns': [
                    r'objects?\s+of\s+(?:the\s+)?company',
                    r'liability.*limited',
                    r'registered\s+office',
                    r'authorized\s+capital'
                ],
                'section_indicators': ['objects', 'liability', 'capital', 'office']
            },
            'incorporation_application': {
                'primary_keywords': [
                    'incorporation application', 'company registration', 'registration authority',
                    'proposed company name', 'registered address'
                ],
                'secondary_keywords': [
                    'initial directors', 'company formation', 'business license',
                    'registration fee', 'application form'
                ],
                'legal_patterns': [
                    r'application\s+for\s+incorporation',
                    r'proposed\s+(?:company\s+)?name',
                    r'registration\s+authority',
                    r'initial\s+directors'
                ],
                'section_indicators': ['application', 'registration', 'directors', 'address']
            },
            'board_resolution': {
                'primary_keywords': [
                    'board resolution', 'directors resolution', 'board meeting',
                    'resolved that', 'unanimously resolved'
                ],
                'secondary_keywords': [
                    'board unanimously', 'meeting minutes', 'director decision',
                    'corporate resolution', 'board approval'
                ],
                'legal_patterns': [
                    r'resolved\s+that',
                    r'board\s+resolution',
                    r'directors?\s+resolve',
                    r'unanimously\s+resolved'
                ],
                'section_indicators': ['resolved', 'meeting', 'board', 'directors']
            },
            'register_members_directors': {
                'primary_keywords': [
                    'register of members', 'register of directors', 'membership register',
                    'shareholder register', 'director details'
                ],
                'secondary_keywords': [
                    'member details', 'share certificates', 'shareholding',
                    'director information', 'member list'
                ],
                'legal_patterns': [
                    r'register\s+of\s+(?:members|directors)',
                    r'member\s+(?:name|details)',
                    r'shares?\s+held',
                    r'director\s+information'
                ],
                'section_indicators': ['register', 'members', 'directors', 'shares']
            },
            'employment_contract': {
                'primary_keywords': [
                    'employment contract', 'employment agreement', 'service agreement',
                    'terms of employment', 'employment terms'
                ],
                'secondary_keywords': [
                    'salary', 'working hours', 'termination', 'probation',
                    'benefits', 'leave entitlement', 'job description'
                ],
                'legal_patterns': [
                    r'employment\s+(?:contract|agreement)',
                    r'terms\s+of\s+employment',
                    r'salary.*payable',
                    r'working\s+hours',
                    r'termination\s+of\s+employment'
                ],
                'section_indicators': ['employment', 'salary', 'hours', 'termination']
            },
        }

    def extract_features(self, document_content):
        full_text = document_content.get('content', {}).get('full_text', '')
        paragraphs = document_content.get('structure', {}).get('paragraphs', [])
        sections = document_content.get('structure', {}).get('sections', [])
        features = {
            'full_text': full_text,
            'word_count': document_content.get('statistics', {}).get('word_count', 0),
            'paragraph_count': len(paragraphs),
            'section_count': len(sections),
            'has_title': self._has_document_title(paragraphs),
            'has_sections': len(sections) > 0,
            'has_numbered_sections': self._has_numbered_sections(sections),
            'has_tables': document_content.get('statistics', {}).get('table_count', 0) > 0,
        }
        return features

    def classify_document(self, document_content, method: str = 'ensemble'):
        features = self.extract_features(document_content)
        if method == 'rule_based':
            return self._classify_rule_based(features)
        elif method == 'ml' and self.is_trained:
            return self._classify_ml(features)
        elif method == 'ensemble':
            return self._classify_ensemble(features)
        else:
            return self._classify_rule_based(features)

    def _classify_rule_based(self, features):
        text = features.get('full_text', '').lower()
        scores = {}
        for doc_type, patterns in self.adgm_patterns.items():
            score = 0.0
            matches = {'primary': [], 'secondary': [], 'patterns': [], 'sections': []}
            for keyword in patterns['primary_keywords']:
                if keyword.lower() in text:
                    score += 3.0
                    matches['primary'].append(keyword)
            for keyword in patterns['secondary_keywords']:
                if keyword.lower() in text:
                    score += 1.5
                    matches['secondary'].append(keyword)
            for pattern in patterns['legal_patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 2.5
                    matches['patterns'].append(pattern)
            for indicator in patterns['section_indicators']:
                if indicator.lower() in text:
                    score += 1.0
                    matches['sections'].append(indicator)
            text_length_factor = len(text.split()) / 1000.0 if len(text.split()) else 1.0
            normalized_score = score / text_length_factor
            scores[doc_type] = {
                'raw_score': score,
                'normalized_score': normalized_score,
                'matches': matches
            }
        if not scores:
            return {
                'predicted_type': 'unknown',
                'confidence': 0.0,
                'method_used': 'rule_based',
                'all_scores': {}
            }
        best_type = max(scores.keys(), key=lambda x: scores[x]['normalized_score'])
        best_score = scores[best_type]['normalized_score']
        sorted_scores = sorted(scores.values(), key=lambda x: x['normalized_score'], reverse=True)
        confidence = min(1.0, best_score / 10.0)
        if len(sorted_scores) > 1:
            score_separation = sorted_scores[0]['normalized_score'] - sorted_scores[1]['normalized_score']
            confidence += min(0.3, score_separation / 5.0)
            confidence = min(1.0, confidence)
        return {
            'predicted_type': best_type,
            'confidence': round(confidence, 3),
            'method_used': 'rule_based',
            'all_scores': {k: v['normalized_score'] for k, v in scores.items()},
            'best_matches': scores[best_type]['matches'],
            'reasoning': f"Matched {len(scores[best_type]['matches']['primary'])} primary keywords, "
                         f"{len(scores[best_type]['matches']['patterns'])} patterns"
        }

    def _classify_ml(self, features):
        # To use rule-based as default until ML is trained
        return self._classify_rule_based(features)

    def _classify_ensemble(self, features):
        rule_result = self._classify_rule_based(features)
        rule_result['method_used'] = 'ensemble'
        confidence = rule_result['confidence']
        matches = rule_result.get('best_matches', {})
        evidence_types = sum(1 for match_type in matches.values() if match_type)
        if evidence_types >= 3:
            confidence = min(1.0, confidence * 1.2)
        rule_result['confidence'] = round(confidence, 3)
        rule_result['evidence_types'] = evidence_types
        return rule_result

    def get_classification_explanation(self, classification_result):
        predicted_type = classification_result.get('predicted_type', 'unknown')
        confidence = classification_result.get('confidence', 0.0)
        method = classification_result.get('method_used', 'unknown')
        explanation = f"Document classified as '{predicted_type.replace('_', ' ').title()}' "
        explanation += f"with {confidence:.1%} confidence using {method} method.\n"
        if 'best_matches' in classification_result:
            matches = classification_result['best_matches']
            if matches.get('primary'):
                explanation += f"Primary indicators found: {', '.join(matches['primary'][:3])}\n"
            if matches.get('patterns'):
                explanation += f"Legal patterns matched: {len(matches['patterns'])} patterns\n"
            if 'reasoning' in classification_result:
                explanation += f"Reasoning: {classification_result['reasoning']}"
        return explanation

    def _has_document_title(self, paragraphs):
        return bool(paragraphs and (paragraphs[0].get('is_header', False) or paragraphs[0].get('content_type','') == 'title_header'))

    def _has_numbered_sections(self, sections):
        return any(re.match(r'^\d+\.', section.get('title', '')) for section in sections)

def classify_document(document_content, method: str = 'ensemble'):
    classifier = DocumentClassifier()
    return classifier.classify_document(document_content, method)

class ADGMKnowledgeBase:
    def __init__(self, openai_api_key=None):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) if openai_api_key else None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = None

    def build_knowledge_base(self, list_of_texts):
        if not self.embeddings:
            raise Exception("OpenAI API key not provided for embeddings")
        chunks = []
        for txt in list_of_texts:
            chunks.extend(self.text_splitter.split_text(txt))
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
        return True

    def retrieve(self, query, k=3):
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [d.page_content for d in docs]

class ComplianceChecker:
    def __init__(self):
        self.required_types = [
            "articles_of_association",
            "memorandum_of_association",
            "incorporation_application",
            "register_members",
            "board_resolution"
        ]
    def check(self, types_list):
        missing = [t for t in self.required_types if t not in types_list]
        return {
            "status": "Complete" if not missing else "Incomplete",
            "missing_documents": missing
        }

class RedFlagDetector:
    def __init__(self):
        self.patterns = [
            {
                "pattern": r"UAE Federal Courts",
                "issue": "Incorrect jurisdiction reference",
                "severity": "High",
                "suggestion": "Replace with ADGM Courts"
            },
            {
                "pattern": r"without\s+ADGM\s+approval",
                "issue": "Missing ADGM approval requirement",
                "severity": "Medium",
                "suggestion": "Add explicit ADGM approval clause"
            }
        ]
    def scan(self, text):
        flags = []
        for pat in self.patterns:
            if re.search(pat["pattern"], text, re.IGNORECASE):
                flags.append(pat)
        return flags

class DocxAnnotator:
    def add_comments(self, original_file_path, paragraphs_and_comments):
        doc = DocxDocument(original_file_path)
        for pc in paragraphs_and_comments:
            para_idx = pc['paragraph_index']
            comment = pc['comment']
            if para_idx < len(doc.paragraphs):
                paragraph = doc.paragraphs[para_idx]
                for run in paragraph.runs:
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                paragraph.add_run(f" [Issue: {comment}]").italic = True
        marked_dir = 'marked_documents'
        if not os.path.exists(marked_dir):
            os.makedirs(marked_dir)
        marked_file = os.path.join(marked_dir, os.path.splitext(os.path.basename(original_file_path))[0] + "_REVIEWED.docx")
        doc.save(marked_file)
        return marked_file

