import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import time

# Streamlit 페이지 설정
st.set_page_config(page_title="IT News Summarizer", page_icon="📰", layout="wide")

# 뉴스 링크를 가져오는 함수
def get_news_links():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    try:
        response = requests.get("https://news.naver.com/section/105", headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # IT/과학 섹션의 기사 컨테이너 찾기
        articles = soup.find_all("div", class_="sa_text")
        news_links = []
        seen_urls = set()
        for article in articles:
            link = article.find("a", href=re.compile(r'n\.news\.naver\.com/mnews/article/\d+/\d+'))
            title_tag = article.find("strong", class_="sa_text_strong")
            if link and title_tag:
                href = link.get("href")
                if href and href not in seen_urls:
                    if not href.startswith("http"):
                        href = "https://news.naver.com" + href
                    title = title_tag.get_text(strip=True) or "제목 없음"
                    st.write(f"디버깅: 링크={href}, 제목={title}")
                    news_links.append({"title": title, "url": href})
                    seen_urls.add(href)
        st.write(f"디버깅: {len(news_links)}개의 뉴스 링크를 찾았습니다.")
        return news_links[:5]  # 최대 5개 기사
    except Exception as e:
        st.error(f"뉴스 링크를 가져오는 중 오류 발생: {e}")
        return []

# 뉴스 본문을 가져오는 함수
def get_article_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find("article", {"id": "dic_area"})
        return content.get_text(strip=True) if content else ""
    except Exception as e:
        st.error(f"기사 본문을 가져오는 중 오류 발생: {e}")
        return ""

# 요약 함수
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="gogamza/kobart-summarization")

def summarize_text(text):
    try:
        summarizer = get_summarizer()
        max_input_length = 512
        summary = summarizer(text[:max_input_length], max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"요약 중 오류 발생: {e}")
        return ""

# 메인 앱
def main():
    st.title("📰 IT 뉴스 요약기")
    st.markdown("네이버 IT/과학 뉴스를 자동으로 가져와 AI로 요약합니다.")

    # 뉴스 링크 가져오기
    with st.spinner("뉴스 목록을 가져오는 중..."):
        news_links = get_news_links()

    if not news_links:
        st.warning("가져온 뉴스가 없습니다. 나중에 다시 시도해주세요.")
        return

    # 뉴스 목록 표시
    for news in news_links:
        with st.expander(f"🗞️ {news['title']}"):
            st.write(f"[원문 읽기]({news['url']})")
            with st.spinner("기사를 요약하는 중..."):
                article = get_article_content(news['url'])
                if article:
                    summary = summarize_text(article)
                    if summary:
                        st.markdown("### 요약")
                        st.write(summary)
                    else:
                        st.warning("요약을 생성할 수 없습니다.")
                else:
                    st.warning("기사 본문을 가져올 수 없습니다.")

if __name__ == "__main__":
    main()
