
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

warnings.filterwarnings('ignore')

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP for semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_NLP = True
except ImportError:
    ADVANCED_NLP = False
    print("⚠️ Install sentence-transformers for semantic features")

@dataclass
class ResearchPaper:
    """Research paper data structure"""
    title: str
    authors: List[str]
    abstract: str
    doi: str
    publication_date: str
    journal: str
    url: str
    citations: int = 0
    keywords: List[str] = field(default_factory=list)
    source: str = ""
    pdf_url: str = ""

class EnhancedResearchAgent:
    """100% FREE Academic Research Agent - NO PAID APIs"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # ALL FREE APIs - NO AUTHENTICATION REQUIRED
        self.apis = {
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'arxiv': 'http://export.arxiv.org/api/query',
            'openalex': 'https://api.openalex.org/works',  # BEST FREE API
            'core': 'https://core.ac.uk:443/api-v2/articles/search/',
            'crossref': 'https://api.crossref.org/works',  # Free metadata
        }

        self.cache = {}
        self.embedder = None

        if ADVANCED_NLP:
            try:
                print("🤖 Loading semantic model...")
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic analysis ready")
            except:
                pass

        print("=" * 70)
        print("🎓 ENHANCED ACADEMIC RESEARCH AGENT")
        print("=" * 70)
        print("📚 FREE Databases: PubMed | ArXiv | OpenAlex | CORE | CrossRef")
        print("🚀 Features: Parallel Search | Deduplication | NLP | Visualizations")
        print("💰 Cost: $0.00 - Everything is FREE!")
        print("=" * 70)

    def search_pubmed(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """PubMed - Medical & Life Sciences - FREE"""
        papers = []
        try:
            # Search IDs
            search_url = f"{self.apis['pubmed']}esearch.fcgi"
            response = self.session.get(search_url, params={
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }, timeout=15)

            pmids = response.json().get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return papers

            # Get details
            detail_url = f"{self.apis['pubmed']}esummary.fcgi"
            details = self.session.get(detail_url, params={
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'json'
            }, timeout=15).json()

            # Get abstracts
            abstract_url = f"{self.apis['pubmed']}efetch.fcgi"
            abstract_response = self.session.get(abstract_url, params={
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }, timeout=15)

            abstracts = {}
            try:
                root = ET.fromstring(abstract_response.content)
                for article in root.findall('.//PubmedArticle'):
                    pmid = article.find('.//PMID').text
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    if abstract_elem is not None and abstract_elem.text:
                        abstracts[pmid] = abstract_elem.text
            except:
                pass

            # Build papers
            for pmid in pmids:
                if pmid in details.get('result', {}):
                    data = details['result'][pmid]
                    authors = [a['name'] for a in data.get('authors', [])[:5] if 'name' in a]

                    papers.append(ResearchPaper(
                        title=data.get('title', ''),
                        authors=authors,
                        abstract=abstracts.get(pmid, ''),
                        doi=data.get('elocationid', '').replace('doi: ', ''),
                        publication_date=data.get('pubdate', ''),
                        journal=data.get('source', 'PubMed'),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        source='PubMed'
                    ))
        except Exception as e:
            print(f"⚠️ PubMed: {e}")

        return papers

    def search_arxiv(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """ArXiv - Physics, CS, Math - FREE"""
        papers = []
        try:
            response = self.session.get(self.apis['arxiv'], params={
                'search_query': f'all:{query}',
                'max_results': max_results,
                'sortBy': 'relevance'
            }, timeout=15)

            root = ET.fromstring(response.content)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                url = entry.find('{http://www.w3.org/2005/Atom}id').text

                authors = [a.find('{http://www.w3.org/2005/Atom}name').text
                          for a in entry.findall('{http://www.w3.org/2005/Atom}author')[:5]]

                keywords = [c.get('term') for c in entry.findall('{http://www.w3.org/2005/Atom}category')]

                pdf_url = url.replace('abs', 'pdf') + '.pdf'

                papers.append(ResearchPaper(
                    title=re.sub(r'\s+', ' ', title.strip()),
                    authors=authors,
                    abstract=re.sub(r'\s+', ' ', summary.strip()),
                    doi='',
                    publication_date=published,
                    journal='arXiv',
                    url=url,
                    keywords=keywords,
                    source='ArXiv',
                    pdf_url=pdf_url
                ))
        except Exception as e:
            print(f"⚠️ ArXiv: {e}")

        return papers

    def search_openalex(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """OpenAlex - ALL FIELDS - FREE, BEST API"""
        papers = []
        try:
            response = self.session.get(self.apis['openalex'], params={
                'search': query,
                'per-page': max_results,
                'mailto': 'research@example.com'
            }, timeout=15)

            results = response.json().get('results', [])

            for r in results:
                # Authors
                authors = [a.get('author', {}).get('display_name', '')
                          for a in r.get('authorships', [])[:5]]

                # Abstract reconstruction
                abstract = ""
                inv_idx = r.get('abstract_inverted_index', {})
                if inv_idx:
                    words = []
                    for word, positions in inv_idx.items():
                        for pos in positions:
                            words.append((pos, word))
                    words.sort()
                    abstract = ' '.join([w for _, w in words])

                # Keywords
                keywords = [c['display_name'] for c in r.get('concepts', [])[:5]]

                # PDF
                pdf_url = r.get('open_access', {}).get('oa_url', '')

                papers.append(ResearchPaper(
                    title=r.get('title', ''),
                    authors=authors,
                    abstract=abstract,
                    doi=r.get('doi', '').replace('https://doi.org/', ''),
                    publication_date=str(r.get('publication_year', '')),
                    journal=r.get('primary_location', {}).get('source', {}).get('display_name', ''),
                    url=r.get('id', ''),
                    citations=r.get('cited_by_count', 0),
                    keywords=keywords,
                    source='OpenAlex',
                    pdf_url=pdf_url
                ))
        except Exception as e:
            print(f"⚠️ OpenAlex: {e}")

        return papers

    def search_core(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """CORE - Open Access - FREE"""
        papers = []
        try:
            response = self.session.get(self.apis['core'], params={
                'q': query,
                'pageSize': min(max_results, 10)
            }, timeout=15)

            for r in response.json().get('data', []):
                authors = [a for a in r.get('authors', []) if a][:5]

                papers.append(ResearchPaper(
                    title=r.get('title', ''),
                    authors=authors,
                    abstract=r.get('description', ''),
                    doi=r.get('doi', ''),
                    publication_date=str(r.get('yearPublished', '')),
                    journal=r.get('publisher', ''),
                    url=r.get('downloadUrl', ''),
                    source='CORE',
                    pdf_url=r.get('downloadUrl', '')
                ))
        except Exception as e:
            print(f"⚠️ CORE: {e}")

        return papers

    def search_all_parallel(self, query: str, max_per_source: int = 10) -> Dict[str, List[ResearchPaper]]:
        """Search ALL databases in PARALLEL - FAST!"""
        print(f"\n🔍 SEARCHING: '{query}'")
        print("=" * 70)

        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.search_pubmed, query, max_per_source): 'PubMed',
                executor.submit(self.search_arxiv, query, max_per_source): 'ArXiv',
                executor.submit(self.search_openalex, query, max_per_source): 'OpenAlex',
                executor.submit(self.search_core, query, max_per_source): 'CORE'
            }

            with tqdm(total=4, desc="Searching", ncols=100) as pbar:
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        papers = future.result()
                        results[source] = papers
                        pbar.set_postfix_str(f"{source}: {len(papers)}")
                    except:
                        results[source] = []
                    pbar.update(1)

        total = sum(len(p) for p in results.values())
        print(f"\n✅ Found {total} papers across {len(results)} databases")
        for source, papers in results.items():
            print(f"   • {source}: {len(papers)}")
        print("=" * 70)

        return results

    def deduplicate(self, papers_by_source: Dict) -> List[ResearchPaper]:
        """Remove duplicates - by DOI and title"""
        all_papers = []
        for papers in papers_by_source.values():
            all_papers.extend(papers)

        seen_dois = set()
        seen_titles = set()
        unique = []

        for paper in all_papers:
            doi = paper.doi.lower() if paper.doi else None
            title = paper.title.lower().strip()

            if doi and doi in seen_dois:
                continue
            if title in seen_titles:
                continue

            if doi:
                seen_dois.add(doi)
            seen_titles.add(title)
            unique.append(paper)

        print(f"\n🔄 Deduplication: {len(all_papers)} → {len(unique)} unique papers")
        return unique

    def analyze(self, papers: List[ResearchPaper]) -> Dict:
        """Advanced statistical analysis"""
        if not papers:
            return {}

        print("\n📊 Analyzing data...")

        analysis = {
            'total': len(papers),
            'sources': Counter([p.source for p in papers]),
            'years': Counter(),
            'authors': Counter(),
            'journals': Counter(),
            'keywords': Counter(),
            'citations': {
                'total': 0,
                'mean': 0,
                'median': 0,
                'max': 0,
                'top_papers': []
            },
            'open_access': 0
        }

        # Years
        for p in papers:
            year = re.search(r'(19|20)\d{2}', str(p.publication_date))
            if year:
                analysis['years'][year.group()] += 1

        # Authors (top 3 per paper)
        for p in papers:
            for author in p.authors[:3]:
                if author:
                    analysis['authors'][author] += 1

        # Journals
        for p in papers:
            if p.journal:
                analysis['journals'][p.journal] += 1

        # Keywords
        for p in papers:
            for kw in p.keywords:
                if kw:
                    analysis['keywords'][kw] += 1

        # Citations
        cites = [p.citations for p in papers if p.citations > 0]
        if cites:
            analysis['citations'] = {
                'total': sum(cites),
                'mean': np.mean(cites),
                'median': np.median(cites),
                'max': max(cites),
                'top_papers': sorted(papers, key=lambda x: x.citations, reverse=True)[:5]
            }

        # Open access
        analysis['open_access'] = sum(1 for p in papers if p.pdf_url)

        print("✅ Analysis complete")
        return analysis

    def semantic_cluster(self, papers: List[ResearchPaper], n_clusters: int = 5):
        """Cluster papers by topic - requires sentence-transformers"""
        if not self.embedder or not papers or len(papers) < n_clusters:
            return None

        print(f"\n🧠 Clustering papers into {n_clusters} topics...")

        try:
            # Embed
            texts = [f"{p.title}. {p.abstract[:300]}" for p in papers]
            embeddings = self.embedder.encode(texts, show_progress_bar=True)

            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            # Group
            clusters = defaultdict(list)
            for i, paper in enumerate(papers):
                clusters[labels[i]].append(paper)

            # Analyze each cluster
            cluster_info = {}
            for cluster_id, cluster_papers in clusters.items():
                all_kw = []
                for p in cluster_papers:
                    all_kw.extend(p.keywords)
                top_kw = Counter(all_kw).most_common(5)

                cluster_info[f"Topic {cluster_id + 1}"] = {
                    'size': len(cluster_papers),
                    'keywords': [k for k, _ in top_kw],
                    'sample_titles': [p.title for p in cluster_papers[:3]]
                }

            print("✅ Clustering complete")
            return cluster_info
        except Exception as e:
            print(f"⚠️ Clustering failed: {e}")
            return None

    def visualize(self, analysis: Dict, papers: List[ResearchPaper]):
        """Create beautiful visualizations"""
        print("\n📈 Creating visualizations...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Publications over time
        ax1 = fig.add_subplot(gs[0, 0])
        if analysis['years']:
            years = sorted(analysis['years'].keys())
            counts = [analysis['years'][y] for y in years]
            ax1.bar(years, counts, color='steelblue', alpha=0.7)
            ax1.set_title('Publications Over Time', fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Papers')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Top authors
        ax2 = fig.add_subplot(gs[0, 1])
        if analysis['authors']:
            top_authors = dict(analysis['authors'].most_common(10))
            ax2.barh(list(top_authors.keys()), list(top_authors.values()), color='coral')
            ax2.set_title('Top 10 Authors', fontweight='bold')
            ax2.set_xlabel('Papers')
            ax2.invert_yaxis()

        # 3. Source distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sources = list(analysis['sources'].keys())
        counts = list(analysis['sources'].values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax3.pie(counts, labels=sources, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Papers by Database', fontweight='bold')

        # 4. Citation distribution
        ax4 = fig.add_subplot(gs[1, 0])
        cites = [p.citations for p in papers if p.citations > 0]
        if cites:
            ax4.hist(cites, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(cites), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cites):.0f}')
            ax4.set_title('Citation Distribution', fontweight='bold')
            ax4.set_xlabel('Citations')
            ax4.set_ylabel('Frequency')
            ax4.legend()

        # 5. Top journals
        ax5 = fig.add_subplot(gs[1, 1])
        if analysis['journals']:
            top_j = dict(analysis['journals'].most_common(8))
            ax5.barh(list(top_j.keys()), list(top_j.values()), color='purple', alpha=0.6)
            ax5.set_title('Top Journals', fontweight='bold')
            ax5.set_xlabel('Papers')
            ax5.invert_yaxis()

        # 6. Keyword cloud
        ax6 = fig.add_subplot(gs[1, 2])
        if analysis['keywords']:
            wc = WordCloud(width=400, height=300, background_color='white', colormap='viridis')
            wc.generate_from_frequencies(dict(analysis['keywords'].most_common(50)))
            ax6.imshow(wc, interpolation='bilinear')
            ax6.axis('off')
            ax6.set_title('Top Keywords', fontweight='bold')

        # 7. Summary stats
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        stats_text = f"""
        📊 RESEARCH SUMMARY

        Total Papers: {analysis['total']}
        Unique Authors: {len(analysis['authors'])}
        Date Range: {min(analysis['years'].keys()) if analysis['years'] else 'N/A'} - {max(analysis['years'].keys()) if analysis['years'] else 'N/A'}

        Total Citations: {analysis['citations']['total']:,}
        Average Citations: {analysis['citations']['mean']:.1f}
        Max Citations: {analysis['citations']['max']}

        Open Access: {analysis['open_access']} papers ({(analysis['open_access']/analysis['total']*100):.1f}%)
        Databases: {', '.join(analysis['sources'].keys())}
        """

        ax7.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('📚 Academic Research Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.show()

        print("✅ Visualizations created")

    def generate_report(self, query: str, papers: List[ResearchPaper], analysis: Dict, clusters: Dict = None) -> str:
        """Generate comprehensive markdown report"""

        report = f"""# 🎓 Academic Research Report

**Topic:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Papers:** {len(papers)}

---

## 📊 Summary

- **Papers Analyzed:** {analysis['total']}
- **Databases:** {', '.join(analysis['sources'].keys())}
- **Total Citations:** {analysis['citations']['total']:,}
- **Average Citations:** {analysis['citations']['mean']:.1f}
- **Open Access:** {analysis['open_access']} ({(analysis['open_access']/analysis['total']*100):.1f}%)

---

## 📅 Publication Timeline

"""
        for year in sorted(analysis['years'].keys(), reverse=True):
            report += f"- **{year}:** {analysis['years'][year]} papers\n"

        report += "\n---\n\n## 👨‍🔬 Top Researchers\n\n"
        for i, (author, count) in enumerate(list(analysis['authors'].most_common(15)), 1):
            report += f"{i}. **{author}** ({count} papers)\n"

        report += "\n---\n\n## 📖 Top Journals\n\n"
        for i, (journal, count) in enumerate(list(analysis['journals'].most_common(10)), 1):
            report += f"{i}. {journal} - {count} papers\n"

        report += "\n---\n\n## 🔑 Top Keywords\n\n"
        for kw, freq in list(analysis['keywords'].most_common(20)):
            report += f"- **{kw}** ({freq})\n"

        if analysis['citations']['top_papers']:
            report += "\n---\n\n## 🌟 Most Cited Papers\n\n"
            for i, paper in enumerate(analysis['citations']['top_papers'], 1):
                report += f"""**{i}. {paper.title}**
- Authors: {', '.join(paper.authors[:3])}
- Citations: {paper.citations:,}
- Journal: {paper.journal}
- URL: {paper.url}

"""

        if clusters:
            report += "\n---\n\n## 🧠 Research Topics (Semantic Clustering)\n\n"
            for topic, info in clusters.items():
                report += f"""### {topic} ({info['size']} papers)
- **Keywords:** {', '.join(info['keywords'])}
- **Sample Papers:**
"""
                for title in info['sample_titles']:
                    report += f"  - {title}\n"
                report += "\n"

        report += f"\n---\n\n## 📚 Sample Papers\n\n"
        for i, paper in enumerate(papers[:10], 1):
            report += f"""**{i}. {paper.title}**
- **Authors:** {', '.join(paper.authors[:3])}
- **Source:** {paper.source}
- **Journal:** {paper.journal}
- **Year:** {re.search(r'(19|20)\d{2}', str(paper.publication_date)).group() if re.search(r'(19|20)\d{2}', str(paper.publication_date)) else 'N/A'}
- **Citations:** {paper.citations:,}
- **URL:** {paper.url}
{f'- **PDF:** {paper.pdf_url}' if paper.pdf_url else ''}
- **Abstract:** {paper.abstract[:200]}...

"""

        report += f"""
---

## 📝 Methodology

- **Search Query:** {query}
- **Databases:** PubMed, ArXiv, OpenAlex, CORE
- **Search Method:** Parallel multi-database retrieval
- **Deduplication:** DOI and title-based
- **Analysis:** Statistical + Semantic (NLP)
- **Cost:** $0.00 (100% Free APIs)

**Generated with Enhanced Academic Research Agent**
"""

        return report

    def save_report(self, report: str, filename: str = None):
        """Save report to file"""
        if not filename:
            filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n💾 Report saved: {filename}")
        return filename

# ============================================================================
# EASY-TO-USE FUNCTIONS
# ============================================================================

def research(query: str, max_per_db: int = 10, visualize: bool = True, cluster: bool = True):
    """
    ONE FUNCTION TO DO EVERYTHING

    Args:
        query: Your research topic
        max_per_db: Papers per database (default 10)
        visualize: Create charts (default True)
        cluster: Semantic clustering (default True)

    Returns:
        report, papers, analysis
    """
    print("\n" + "=" * 70)
    print("🎓 ACADEMIC RESEARCH AGENT - 100% FREE")
    print("=" * 70)

    # Initialize
    agent = EnhancedResearchAgent()

    # Search all databases
    papers_by_source = agent.search_all_parallel(query, max_per_db)

    # Deduplicate
    unique_papers = agent.deduplicate(papers_by_source)

    if not unique_papers:
        print("\n❌ No papers found. Try a different query.")
        return None, None, None

    # Analyze
    analysis = agent.analyze(unique_papers)

    # Cluster (optional)
    clusters = None
    if cluster and ADVANCED_NLP and len(unique_papers) >= 5:
        clusters = agent.semantic_cluster(unique_papers, n_clusters=min(5, len(unique_papers)//2))

    # Visualize (optional)
    if visualize:
        agent.visualize(analysis, unique_papers)

    # Generate report
    report = agent.generate_report(query, unique_papers, analysis, clusters)

    print("\n" + "=" * 70)
    print("✅ RESEARCH COMPLETE!")
    print("=" * 70)
    print(f"📄 {len(unique_papers)} unique papers found")
    print(f"📊 {analysis['citations']['total']:,} total citations")
    print(f"🔓 {analysis['open_access']} open access papers")
    print("=" * 70)

    # Display report
    print("\n" + report)

    return report, unique_papers, analysis

# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

EXAMPLE_TOPICS = [
    "machine learning healthcare",
    "blockchain education",
    "climate change mitigation",
    "quantum computing",
    "CRISPR gene editing",
    "renewable energy",
    "artificial intelligence ethics",
    "cancer immunotherapy"
]

def demo():
    """Run interactive demo"""
    print("\n📚 RESEARCH TOPICS:")
    for i, topic in enumerate(EXAMPLE_TOPICS, 1):
        print(f"  {i}. {topic}")

    choice = input("\nSelect topic (1-8) or enter your own: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_TOPICS):
        topic = EXAMPLE_TOPICS[int(choice) - 1]
    else:
        topic = choice
      

    if not topic:
        print("❌ Invalid input")
        return

    # DO RESEARCH
    report, papers, analysis = research(topic, max_per_db=10)

    # Save option
    save = input("\n💾 Save report? (y/n): ").strip().lower()
    if save == 'y':
        agent = EnhancedResearchAgent()
        filename = agent.save_report(report)
        print(f"✅ Saved to {filename}")
