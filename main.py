import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from pathlib import Path
from docx import Document
import PyPDF2
import chardet
import json
import dateutil.parser
from datetime import datetime
from collections import defaultdict

@dataclass
class MatchConfig:
    skills_weight: float = 0.35
    experience_weight: float = 0.25
    education_weight: float = 0.15
    projects_weight: float = 0.10
    achievements_weight: float = 0.10
    misc_weight: float = 0.05
    embedding_weight: float = 0.6
    semantic_weight: float = 0.3
    exact_weight: float = 0.1
    match_threshold: float = 0.6
    recency_weight: float = 0.7
    skill_depth_weight: float = 0.6
    seniority_match_weight: float = 0.8

@dataclass
class CandidateScore:
    id: str
    total_score: float
    category_scores: Dict[str, float]
    keywords_matched: Dict[str, List[str]]
    skill_depth_scores: Dict[str, float]
    seniority_match: float
    years_experience: float
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'total_score': self.total_score,
            'details': self.category_scores,
            'keywords_matched': self.keywords_matched,
            'skill_details': {k: round(v, 2) for k, v in self.skill_depth_scores.items()},
            'seniority_match': round(self.seniority_match, 2),
            'years_experience': round(self.years_experience, 2)
        }

class CVMatcher:
    def __init__(self, config: Optional[MatchConfig] = None):
        self.config = config or MatchConfig()
        
        self._tokenizer = None
        self._model = None
        self._nlp = None
        
        self.section_headers = {
            'Skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'proficiencies', 'technologies'],
            'Experience': ['experience', 'work experience', 'professional experience', 'employment history', 'career history'],
            'Education': ['education', 'academic background', 'qualifications', 'academic qualifications', 'degrees'],
            'Projects': ['projects', 'key projects', 'significant projects', 'personal projects', 'portfolio'],
            'Achievements': ['achievements', 'awards', 'honors', 'certifications', 'publications', 'patents'],
            'Summary': ['summary', 'professional summary', 'profile', 'objective', 'about me']
        }
        
        self.skill_synonyms = {
            'python': ['py', 'python3'],
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
            'machine learning': ['ml', 'deep learning', 'ai', 'artificial intelligence'],
            'aws': ['amazon web services', 'ec2', 'lambda', 's3'],
            'frontend': ['front-end', 'front end', 'ui', 'user interface'],
            'backend': ['back-end', 'back end', 'server-side'],
            'devops': ['ci/cd', 'continuous integration', 'deployment'],
            'data science': ['data analytics', 'data analysis', 'analytics'],
            'management': ['leadership', 'team lead', 'supervisor']
        }
        
        self.education_terms = {
            'bachelor': ['bs', 'ba', 'b.s.', 'b.a.', 'undergraduate', 'bsc', 'btech', 'b.tech'],
            'master': ['ms', 'ma', 'm.s.', 'm.a.', 'graduate', 'msc', 'mtech', 'm.tech', 'mba', 'm.b.a.'],
            'doctorate': ['phd', 'ph.d.', 'doctoral', 'doctor of philosophy']
        }
        
        self.seniority_levels = {
            'entry': ['entry', 'junior', 'associate', 'intern', 'trainee'],
            'mid': ['mid level', 'intermediate', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'staff'],
            'management': ['manager', 'director', 'head', 'chief', 'vp', 'executive']
        }

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            self._model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        return self._model
    
    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load('en_core_web_sm')
        return self._nlp

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).lower().strip()
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(768)
        cleaned_text = self.clean_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return mean_embedding.numpy()[0]

    def extract_skills(self, text: str) -> Set[str]:
        if not text:
            return set()
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)
        skills = set()
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                skills.add(chunk.text)
        
        for entity in doc.ents:
            if entity.label_ in ['SKILL', 'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART']:
                skills.add(entity.text)
        
        skill_patterns = [
            r'\b(python|java|javascript|js|c\+\+|c#|ruby|php|sql|html|css|react|angular|vue|node|go|swift|rust)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|terraform|github|gitlab|bitbucket)\b',
            r'\b(machine learning|ml|ai|nlp|computer vision|cv|data science|big data|devops|agile|scrum)\b',
            r'\b(django|flask|spring|laravel|express|tensorflow|pytorch|scikit-learn|pandas|numpy)\b'
        ]
        for pattern in skill_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            skills.update(matches)
        
        expanded_skills = set(skills)
        for skill in skills:
            for primary, variations in self.skill_synonyms.items():
                if skill in variations:
                    expanded_skills.add(primary)
                elif skill == primary:
                    expanded_skills.update(variations)
        
        for token in doc:
            if token.pos_ == 'NOUN' and token.is_alpha and len(token.text) > 2:
                skills.add(token.text)
            if token.pos_ == 'ADJ' and token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
                compound = f"{token.text} {token.head.text}"
                if len(compound.split()) <= 3:
                    skills.add(compound)
        
        return skills

    def extract_years_experience(self, text: str) -> List[Tuple[str, float, str]]:
        years_pattern = r'(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years?|yrs?)\s+(?:of\s+)?(?:in|with)?\s+([a-zA-Z0-9#\+\.\s]+)'
        tech_years = []
        matches = re.finditer(years_pattern, text.lower())
        for match in matches:
            years = float(match.group(1))
            tech = match.group(2).strip()
            sentence_pattern = r'[^.!?]*' + re.escape(match.group(0)) + r'[^.!?]*[.!?]'
            context_match = re.search(sentence_pattern, text.lower())
            context = context_match.group(0).strip() if context_match else ""
            tech_years.append((tech, years, context))
        return tech_years

    def extract_dates(self, text: str) -> List[Tuple[datetime, datetime, str]]:
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*\.?\s+(\d{4})\s*(?:-|to|–|—)\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Present|Current)\w*\.?\s*(\d{4})?'
        date_ranges = []
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        for match in matches:
            start_month, start_year = match.group(1), int(match.group(2))
            end_month, end_year_str = match.group(3), match.group(4)
            try:
                start_date = dateutil.parser.parse(f"1 {start_month} {start_year}")
                if end_month.lower() in ['present', 'current']:
                    end_date = datetime.now()
                else:
                    end_year = int(end_year_str) if end_year_str else start_year
                    end_date = dateutil.parser.parse(f"1 {end_month} {end_year}")
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end].strip()
                date_ranges.append((start_date, end_date, context))
            except (ValueError, dateutil.parser.ParserError):
                continue
        return date_ranges

    def calculate_experience_timeline(self, text: str) -> Tuple[float, Dict]:
        date_ranges = self.extract_dates(text)
        date_ranges.sort(key=lambda x: x[0])
        total_years = 0
        timeline = {}
        for i, (start_date, end_date, context) in enumerate(date_ranges):
            duration = (end_date - start_date).days / 365.25
            total_years += duration
        return total_years, timeline

    def detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        level_counts = {level: 0 for level in self.seniority_levels.keys()}
        for level, terms in self.seniority_levels.items():
            for term in terms:
                level_counts[level] += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
        experience_pattern = r'(\d+)(?:\+)?\s+years'
        experience_matches = re.findall(experience_pattern, text_lower)
        years_mentioned = [int(y) for y in experience_matches if y.isdigit()]
        if years_mentioned:
            avg_years = sum(years_mentioned) / len(years_mentioned)
            if avg_years < 3:
                level_counts['entry'] += 2
            elif avg_years < 6:
                level_counts['mid'] += 2
            elif avg_years < 10:
                level_counts['senior'] += 2
            else:
                level_counts['management'] += 2
        management_terms = ['manage', 'lead', 'supervise', 'oversee', 'direct', 'coordinate']
        for term in management_terms:
            if re.search(r'\b' + re.escape(term) + r'\w*\b', text_lower):
                level_counts['management'] += 1
        return max(level_counts.items(), key=lambda x: x[1])[0] if any(level_counts.values()) else 'mid'

    def calculate_skill_depth(self, text: str, skills: Set[str]) -> Dict[str, float]:
        text_lower = text.lower()
        skill_depth = {}
        for skill in skills:
            skill_lower = skill.lower()
            count = len(re.findall(r'\b' + re.escape(skill_lower) + r'\b', text_lower))
            summary_pattern = r'(summary|profile|about me|introduction|objective)[^\n]*?\n+(.{50,500})'
            summary_match = re.search(summary_pattern, text_lower)
            summary_bonus = 0.3 if summary_match and skill_lower in summary_match.group(2).lower() else 0
            expertise_patterns = [
                r'expertise\s+in\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)',
                r'expert\s+in\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)',
                r'proficient\s+in\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)',
                r'advanced\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)',
                r'specialist\s+in\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)'
            ]
            expertise_bonus = 0.4 if any(re.search(pattern, text_lower) for pattern in expertise_patterns) else 0
            years_pattern = r'(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years?|yrs?)\s+(?:of\s+)?(?:in|with)?\s+([^.]*?' + re.escape(skill_lower) + r'[^.]*?)'
            years_match = re.search(years_pattern, text_lower)
            years_bonus = min(0.5, float(years_match.group(1)) / 10) if years_match else 0
            base_score = min(1.0, count / 5)
            skill_depth[skill] = min(1.0, base_score + summary_bonus + expertise_bonus + years_bonus)
        return skill_depth

    def keyword_matching(self, cv_text: str, keywords: List[str], strict: bool = False) -> Tuple[float, List[str]]:
        if not keywords or not cv_text:
            return 0.0, []
        cv_lower = self.clean_text(cv_text)
        matched_keywords = []
        for kw in keywords:
            clean_kw = self.clean_text(kw)
            if not clean_kw:
                continue
            if re.search(r'\b' + re.escape(clean_kw) + r'\b', cv_lower):
                matched_keywords.append(kw)
                continue
            if strict:
                continue
            for primary, variations in self.skill_synonyms.items():
                if clean_kw == primary and any(re.search(r'\b' + re.escape(var) + r'\b', cv_lower) for var in variations):
                    matched_keywords.append(kw)
                    break
                elif clean_kw in variations and re.search(r'\b' + re.escape(primary) + r'\b', cv_lower):
                    matched_keywords.append(kw)
                    break
        score = len(matched_keywords) / len(keywords) if keywords else 0.0
        return score, matched_keywords

    def semantic_matching(self, cv_text: str, jd_text: str) -> float:
        if not cv_text or not jd_text:
            return 0.0
        cv_doc = self.nlp(cv_text[:10000])
        jd_doc = self.nlp(jd_text[:10000])
        cv_sentences = [sent.text for sent in cv_doc.sents][:50]
        jd_sentences = [sent.text for sent in jd_doc.sents][:50]
        if not cv_sentences or not jd_sentences:
            return 0.0
        cv_embeddings = [self.get_embedding(sent) for sent in cv_sentences]
        jd_embeddings = [self.get_embedding(sent) for sent in jd_sentences]
        similarities = [max(cosine_similarity([jd_emb], [cv_emb])[0][0] for cv_emb in cv_embeddings) for jd_emb in jd_embeddings]
        return sum(similarities) / len(similarities) if similarities else 0.0

    def compute_category_score(self, cv_embedding: np.ndarray, jd_embedding: np.ndarray, cv_text: str, jd_text: str, jd_keywords: List[str]) -> Tuple[float, List[str]]:
        emb_similarity = cosine_similarity([cv_embedding], [jd_embedding])[0][0]
        kw_score, matched_keywords = self.keyword_matching(cv_text, jd_keywords)
        sem_similarity = self.semantic_matching(cv_text, jd_text)
        combined_score = (self.config.embedding_weight * emb_similarity + 
                         self.config.exact_weight * kw_score + 
                         self.config.semantic_weight * sem_similarity)
        return combined_score, matched_keywords

    def parse_sections(self, text: str, section_type: str) -> Dict[str, str]:
        if not text:
            return {category: "" for category in self.section_headers}
        sections = {category: "" for category in self.section_headers}
        lines = text.split('\n')
        section_start_indices = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            line_lower = line.lower()
            for category, headers in self.section_headers.items():
                for header in headers:
                    if (re.search(r'\b' + re.escape(header) + r'\b', line_lower) or 
                        header == line_lower or 
                        line_lower.startswith(header + ":") or 
                        line_lower.startswith(header + " ")):
                        if len(line) < 30 or line.isupper() or line[0].isupper():
                            section_start_indices[category] = i
                            break
        sorted_sections = sorted(section_start_indices.items(), key=lambda x: x[1])
        for i, (category, start_idx) in enumerate(sorted_sections):
            end_idx = len(lines) if i == len(sorted_sections) - 1 else sorted_sections[i+1][1]
            section_lines = lines[start_idx+1:end_idx]
            sections[category] = '\n'.join(section_lines).strip()
        empty_sections = [k for k, v in sections.items() if not v or len(v) < 20]
        if empty_sections:
            for category in empty_sections:
                headers = self.section_headers[category]
                for header in headers:
                    header_pattern = rf"(?i)(?:^|\n)[\s]*{re.escape(header)}[\s:]*(?:\n|$)(.*?)(?=(?:^|\n)[\s]*(?:{'|'.join([re.escape(h) for cat in self.section_headers for h in self.section_headers[cat] if cat != category])})|$)"
                    matches = re.findall(header_pattern, text, re.DOTALL)
                    if matches:
                        sections[category] = max(matches, key=len).strip()
                        break
        for section in ['Experience', 'Education']:
            if not sections[section]:
                if section == 'Experience':
                    work_patterns = [
                        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*(?:-|–|to)\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present)',
                        r'(?:20|19)\d{2}\s*(?:-|–|to)\s*(?:20|19)\d{2}|Present',
                        r'\b(?:senior|lead|principal|junior|associate)?\s*([a-z\s]+(?:engineer|developer|designer|manager|analyst|specialist|director|consultant|architect))'
                    ]
                    for pattern in work_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            match_text = match if isinstance(match, str) else match[0]
                            match_idx = text.lower().find(match_text.lower())
                            if match_idx >= 0:
                                context_start = max(0, text.rfind('\n', 0, match_idx))
                                context_end = text.find('\n\n', match_idx) if text.find('\n\n', match_idx) >= 0 else len(text)
                                sections[section] += text[context_start:context_end] + "\n\n"
                    sections[section] = re.sub(r'\n{3,}', '\n\n', sections[section]).strip()
                elif section == 'Education':
                    edu_patterns = [
                        r'(?:bachelor|master|phd|doctorate|mba|bs|ms|ba|ma|b\.s\.|m\.s\.|ph\.d\.)[^\n.]*(?:degree|of|in)[^\n.]*',
                        r'(?:university|college|institute|school)[^\n.]*',
                        r'\b(?:GPA|grade)[^\n.]*\d+\.\d+'
                    ]
                    for pattern in edu_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            match_idx = text.lower().find(match.lower())
                            if match_idx >= 0:
                                context_start = max(0, text.rfind('\n', 0, match_idx))
                                context_end = text.find('\n\n', match_idx) if text.find('\n\n', match_idx) >= 0 else len(text)
                                sections[section] += text[context_start:context_end] + "\n\n"
                    sections[section] = re.sub(r'\n{3,}', '\n\n', sections[section]).strip()
        return sections

    def preprocess_document(self, file_path: str) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        try:
            if file_path.suffix.lower() == '.docx':
                doc = Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            elif file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    return '\n'.join([page.extract_text() for page in pdf_reader.pages])
            else:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
        except Exception:
            return ""

    def match_cv_to_jd(self, cv_text: str, jd_text: str, candidate_id: str = "unknown") -> CandidateScore:
        if not cv_text or not jd_text:
            return CandidateScore(id=candidate_id, total_score=0.0, category_scores={}, keywords_matched={},
                                skill_depth_scores={}, seniority_match=0.0, years_experience=0.0)
        
        cv_sections = self.parse_sections(cv_text, "CV")
        jd_sections = self.parse_sections(jd_text, "JD")
        cv_skills = self.extract_skills(cv_text)
        jd_skills = self.extract_skills(jd_text)
        skill_depth = self.calculate_skill_depth(cv_text, cv_skills.intersection(jd_skills))
        years_experience, _ = self.calculate_experience_timeline(cv_text)
        
        cv_seniority = self.detect_seniority(cv_text)
        jd_seniority = self.detect_seniority(jd_text)
        seniority_levels_ordered = {'entry': 1, 'mid': 2, 'senior': 3, 'management': 4}
        cv_level = seniority_levels_ordered.get(cv_seniority, 2)
        jd_level = seniority_levels_ordered.get(jd_seniority, 2)
        seniority_match = max(0, 1.0 - (abs(cv_level - jd_level) * 0.3))
        
        category_scores = {}
        all_keywords_matched = {}
        
        total_applied_weight = 0.0
        
        for category, weight_attr in [
            ('Skills', 'skills_weight'), ('Experience', 'experience_weight'), ('Education', 'education_weight'),
            ('Projects', 'projects_weight'), ('Achievements', 'achievements_weight'), ('Summary', 'misc_weight')
        ]:
            weight = getattr(self.config, weight_attr)
            total_applied_weight += weight
            
            cv_category_text = cv_sections.get(category, "")
            jd_category_text = jd_sections.get(category, "")
            
            cv_embedding = self.get_embedding(cv_category_text)
            jd_embedding = self.get_embedding(jd_category_text)
            jd_keywords = list(self.extract_skills(jd_category_text))
            score, matched_keywords = self.compute_category_score(cv_embedding, jd_embedding, cv_category_text, jd_category_text, jd_keywords)
            category_scores[category] = score * weight
            if matched_keywords:
                all_keywords_matched[category] = matched_keywords
        
        if 'Experience' in category_scores:
            experience_score = category_scores['Experience']
            adjustment_factors = 1.0
            
            if years_experience > 0:
                experience_bonus = min(1.0, years_experience / 10.0)
                experience_score = max(experience_score, experience_bonus * self.config.experience_weight)
            
            adjustment_factors += (seniority_match * self.config.seniority_match_weight)
            
            adjustment_factors = min(2.0, adjustment_factors)
            category_scores['Experience'] = experience_score * adjustment_factors
        
        if 'Skills' in category_scores and skill_depth and self.config.skill_depth_weight > 0:
            avg_depth = sum(skill_depth.values()) / len(skill_depth) if skill_depth else 0
            skill_depth_adjustment = min(1.0, 1 + (avg_depth * self.config.skill_depth_weight))
            category_scores['Skills'] *= skill_depth_adjustment
        
        unnormalized_score = sum(category_scores.values())
        total_score = min(1.0, unnormalized_score / total_applied_weight)
        
        return CandidateScore(
            id=candidate_id,
            total_score=total_score,
            category_scores=category_scores,
            keywords_matched=all_keywords_matched,
            skill_depth_scores=skill_depth,
            seniority_match=seniority_match,
            years_experience=years_experience
        )

    def batch_process(self, cv_dir: str, job_description: str, output_file: str = "match_results.json") -> List[CandidateScore]:
        cv_dir = Path(cv_dir)
        if not cv_dir.is_dir():
            return []
        jd_text = self.preprocess_document(job_description) if Path(job_description).is_file() else job_description
        if not jd_text:
            return []
        results = []
        cv_files = list(cv_dir.glob("*.pdf")) + list(cv_dir.glob("*.docx")) + list(cv_dir.glob("*.txt"))
        for cv_file in cv_files:
            try:
                candidate_id = cv_file.stem
                cv_text = self.preprocess_document(cv_file)
                if not cv_text:
                    continue
                score = self.match_cv_to_jd(cv_text, jd_text, candidate_id)
                results.append(score)
            except Exception:
                pass
        results.sort(key=lambda x: x.total_score, reverse=True)
        with open(output_file, 'w') as f:
            json.dump({"job_description": job_description, "processed_at": datetime.now().isoformat(),
                      "candidates": [r.to_dict() for r in results]}, f, indent=2)
        return results