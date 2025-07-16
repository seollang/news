import streamlit as st
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from transformers import pipeline
import re

# Streamlit 페이지 설정
st.set_page_config(page_title="IT News Summarizer", page_icon="📰", layout="wide")

# 뉴스 링크를 가져오는 함수
@st.cache_data(ttl=3600)  # 1시간 캐싱
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
                    news_links.append({"title": title, "url": href})
                    seen_urls.add(href)
        return news_links[:5]  # 최대 5개 기사
    except Exception as e:
        st.error(f"뉴스 링크를 가져오는 중 오류 발생: {e}")
        return []

# 뉴스 본문을 비동기적으로 가져오는 함수
@st.cache_data(ttl=3600)
async def get_article_content_async(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                article = soup.find("article", {"id": "dic_area"})
                return article.get_text(strip=True) if article else ""
    except Exception as e:
        st.error(f"기사 본문을 가져오는 중 오류 발생: {e}")
        return ""

# 요약 함수
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="ainize/kobart-news")

@st.cache_data(ttl=3600)
def summarize_text(_text):
    try:
        summarizer = get_summarizer()
        max_input_length = 512
        summary = summarizer(_text[:max_input_length], max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"요약 중 오류 발생: {e}")
        return ""

# 메인 앱
def main():
    st.title("📰 IT 뉴스 요약기")
    st.markdown("네이버 IT/과학 뉴스를 선택하여 AI로 요약합니다.")

    # 뉴스 링크 가져오기
    with st.spinner("뉴스 목록을 가져오는 중..."):
        news_links = get_news_links()

    if not news_links:
        st.warning("가져온 뉴스가 없습니다. 나중에 다시 시도해주세요.")
        return

    # 뉴스 선택 드롭다운
    selected_news = st.selectbox(
        "요약할 뉴스를 선택하세요:",
        options=news_links,
        format_func=lambda x: x['title']
    )

    if selected_news:
        st.write(f"[원문 읽기]({selected_news['url']})")
        with st.spinner("기사를 요약하는 중..."):
            # 비동기적으로 기사 본문 가져오기
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            article = loop.run_until_complete(get_article_content_async(selected_news['url']))
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
