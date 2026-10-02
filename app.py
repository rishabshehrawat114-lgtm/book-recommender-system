"""Book Recommendation System - User-Facing Web Application.

A clean, modern, and consumer-friendly book recommendation interface
powered by the underlying hybrid recommendation service.
"""

import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.artifacts import load_recommendation_service

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clean Consumer UI Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Layout */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    .feature-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem;
        height: 100%;
        text-align: left;
    }
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Book Cards */
    .book-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .book-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
    }
    .book-title {
        font-weight: 600;
        font-size: 0.98rem;
        color: #0F172A;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
        line-height: 1.35;
    }
    .book-author {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 0.35rem;
    }
    .book-meta {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    .book-reason {
        background-color: #F1F5F9;
        color: #334155;
        border-radius: 6px;
        padding: 0.4rem 0.6rem;
        font-size: 0.8rem;
        line-height: 1.35;
        margin-top: auto;
    }
    .poster-placeholder {
        width: 100%;
        height: 220px;
        background-color: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 6px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #64748B;
        font-size: 0.85rem;
        text-align: center;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Recommendation Service Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing book recommendations...")
def get_service(artifacts_dir: str = "artifacts"):
    """Load serialized models and construct RecommendationService once."""
    abs_artifacts = os.path.join(REPO_ROOT, artifacts_dir)
    return load_recommendation_service(abs_artifacts, default_alpha=0.5)


def sanitize_poster_url(url: Any) -> Optional[str]:
    """Ensure poster URL uses HTTPS and is valid."""
    if not url or pd.isna(url):
        return None
    url_str = str(url).strip()
    if url_str.startswith("http://"):
        url_str = url_str.replace("http://", "https://")
    if url_str.startswith("https://"):
        return url_str
    return None


def render_book_card(col, rank: int, item: Dict[str, Any], context_reason: str = ""):
    """Render a clean, uncluttered book recommendation card."""
    with col:
        with st.container(border=True):
            img_url = sanitize_poster_url(item.get("img_url"))
            if img_url:
                try:
                    st.image(img_url, use_container_width=True)
                except Exception:
                    st.markdown('<div class="poster-placeholder">📖<br>Cover Not Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="poster-placeholder">📖<br>Cover Not Available</div>', unsafe_allow_html=True)

            title = item.get("title", "Unknown Title")
            st.markdown(f"**{title}**")

            author = item.get("author") or "Unknown Author"
            st.markdown(f"*by {author}*")

            pub = item.get("publisher")
            year = item.get("year")
            meta_items = []
            if pub and str(pub) != "nan":
                meta_items.append(str(pub))
            if year and str(year) != "nan":
                meta_items.append(str(year))
            if meta_items:
                st.caption(" • ".join(meta_items))

            reason = context_reason
            if not reason:
                strat = item.get("strategy")
                if strat == "popularity_fallback":
                    reason = "Popular choice among readers"
                elif strat == "seed_content_fallback":
                    reason = "Matches books you selected"
                else:
                    reason = "Recommended based on reader preferences"

            st.info(f"💡 {reason}")


def render_recommendation_grid(recs: List[Dict[str, Any]], default_reason: str = "", num_cols: int = 3):
    """Render recommendations in a balanced multi-column grid."""
    if not recs:
        return
    for row_idx in range(0, len(recs), num_cols):
        row_recs = recs[row_idx : row_idx + num_cols]
        cols = st.columns(num_cols)
        for col_idx, rec in enumerate(row_recs):
            render_book_card(
                cols[col_idx],
                row_idx + col_idx + 1,
                rec,
                context_reason=default_reason,
            )


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Session state for seamless navigation
    if "nav_section" not in st.session_state:
        st.session_state.nav_section = "Home"

    # Sidebar Navigation
    st.sidebar.title("📚 Book Recommender")
    nav_selection = st.sidebar.radio(
        "Navigation",
        ["Home", "Recommend Books", "About"],
        index=["Home", "Recommend Books", "About"].index(st.session_state.nav_section),
    )
    st.session_state.nav_section = nav_selection

    # Load service
    try:
        service, ratings_df = get_service()
        catalog_books = sorted(service.content_recommender.get_supported_books())
    except FileNotFoundError as e:
        st.error(f"⚠️ Recommendation data not ready: {e}")
        st.info("Please ensure the model training scripts have finished creating artifacts.")
        return
    except Exception as e:
        st.error(f"⚠️ Unable to load recommender: {e}")
        return

    # =========================================================================
    # SECTION 1: HOME
    # =========================================================================
    if st.session_state.nav_section == "Home":
        st.markdown('<div class="hero-title">Book Recommendation System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Discover books you may enjoy based on your reading preferences and similar books.</div>',
            unsafe_allow_html=True,
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("✨ Get Book Recommendations", type="primary", use_container_width=True):
                st.session_state.nav_section = "Recommend Books"
                st.rerun()

        st.markdown("---")
        st.markdown("### How It Helps You Find Your Next Read")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Find Similar Books</div>
                    <div class="feature-desc">Enter any book you love, and discover titles with similar storylines, themes, and author styles.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Personalized Picks</div>
                    <div class="feature-desc">Select your reading profile to get suggestions tailored specifically to your past reading taste.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🌱</div>
                    <div class="feature-title">New Reader Friendly</div>
                    <div class="feature-desc">Brand new here? Pick a couple of favorite titles to get instant suggestions, or explore our top-rated catalog books.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # SECTION 2: RECOMMEND BOOKS
    # =========================================================================
    elif st.session_state.nav_section == "Recommend Books":
        st.markdown('<div class="hero-title">Find Books to Read</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Choose how you would like to discover your next favorite book.</div>', unsafe_allow_html=True)

        mode = st.radio(
            "What kind of recommendations are you looking for?",
            ["Similar Books", "Personalized Recommendations", "New User"],
            horizontal=True,
        )
        st.markdown("---")

        # ---------------------------------------------------------------------
        # Mode 1: Similar Books
        # ---------------------------------------------------------------------
        if mode == "Similar Books":
            st.subheader("Similar Books")
            st.caption("Select a book you enjoyed to discover other books with a similar feel and author style.")

            default_idx = catalog_books.index("1984") if "1984" in catalog_books else 0
            selected_book = st.selectbox(
                "Search or choose a book:",
                catalog_books,
                index=default_idx,
            )

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                get_recs = st.button("Recommend Similar Books", type="primary", use_container_width=True)

            if get_recs:
                with st.spinner("Finding books similar to your selection..."):
                    try:
                        recs = service.recommend_for_book(
                            selected_book,
                            n_recommendations=6,
                            method="hybrid",
                        )
                        if not recs:
                            st.info("No recommendations found for this book. Try selecting another title.")
                        else:
                            st.success(f"Books similar to **{selected_book}**:")
                            render_recommendation_grid(
                                recs,
                                default_reason=f"Recommended because it is similar to books you selected.",
                                num_cols=3,
                            )
                    except Exception as e:
                        st.error("Unable to generate recommendations. Please try another title.")

        # ---------------------------------------------------------------------
        # Mode 2: Personalized Recommendations
        # ---------------------------------------------------------------------
        elif mode == "Personalized Recommendations":
            st.subheader("Personalized Recommendations")
            st.caption("Get tailored reading recommendations based on your reading history.")

            # Filter active readers from dataset
            explicit_ratings = ratings_df[ratings_df["rating"] > 0]
            top_user_counts = explicit_ratings["user_id"].value_counts().head(20)
            sample_uids = list(top_user_counts.index)

            c_pick, c_custom = st.columns(2)
            with c_pick:
                selected_sample = st.selectbox(
                    "Select your profile:",
                    sample_uids,
                    format_func=lambda uid: f"Reader Profile #{uid}",
                )
            with c_custom:
                custom_uid = st.text_input("Or enter a Reader ID:", placeholder="e.g. 277427")

            active_uid = int(custom_uid) if custom_uid.strip().isdigit() else selected_sample

            user_history = ratings_df[ratings_df["user_id"] == active_uid]
            if user_history.empty:
                st.warning(f"Profile #{active_uid} was not found. Please pick from the list or enter a valid ID.")
            else:
                n_books = len(user_history)
                st.caption(f"Profile #{active_uid} has rated **{n_books}** books in the catalog.")

                with st.expander("📚 Books you've recently rated"):
                    st.dataframe(
                        user_history[["title", "author", "rating"]].head(8).reset_index(drop=True),
                        use_container_width=True,
                    )

                col_btn, _ = st.columns([1, 4])
                with col_btn:
                    get_user_recs = st.button("Get Personalized Recommendations", type="primary", use_container_width=True)

                if get_user_recs:
                    with st.spinner("Finding books matching your taste..."):
                        u_df = user_history[["title", "rating"]]
                        recs = service.recommend_for_user(u_df, n_recommendations=6)

                        if not recs:
                            st.info("No recommendations found for this profile.")
                        else:
                            st.success("Based on your reading history:")
                            render_recommendation_grid(
                                recs,
                                default_reason="Based on your reading history",
                                num_cols=3,
                            )

        # ---------------------------------------------------------------------
        # Mode 3: New User
        # ---------------------------------------------------------------------
        elif mode == "New User":
            st.subheader("New Reader Recommendations")
            st.caption("Choose a few books you like to get personalized suggestions, or discover popular reads.")

            seed_selection = st.multiselect(
                "Choose a few books you like:",
                catalog_books,
                default=[catalog_books[0]] if catalog_books else [],
                placeholder="Search and pick one or more books...",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                get_seed_recs = st.button("Get Recommendations", type="primary", use_container_width=True)
            with col2:
                get_popular_recs = st.button("Discover Popular Books Instead", use_container_width=True)

            if get_seed_recs:
                if not seed_selection:
                    # Graceful fallback if user clicked without picking any books
                    with st.spinner("Retrieving popular catalog titles..."):
                        recs = service.recommend_for_new_user(n_recommendations=6)
                        st.info("Here are popular books to get you started:")
                        render_recommendation_grid(
                            recs,
                            default_reason="Popular choice among readers",
                            num_cols=3,
                        )
                else:
                    with st.spinner("Finding recommendations from your favorites..."):
                        recs = service.recommend_for_new_user(seed_books=seed_selection, n_recommendations=6)
                        st.success("Recommendations based on your selected favorites:")
                        render_recommendation_grid(
                            recs,
                            default_reason="Recommended because it is similar to books you selected",
                            num_cols=3,
                        )

            elif get_popular_recs:
                with st.spinner("Retrieving popular catalog titles..."):
                    recs = service.recommend_for_new_user(n_recommendations=6)
                    st.success("Popular books among catalog readers:")
                    render_recommendation_grid(
                        recs,
                        default_reason="Popular choice among readers",
                        num_cols=3,
                    )

    # =========================================================================
    # SECTION 3: ABOUT
    # =========================================================================
    elif st.session_state.nav_section == "About":
        st.markdown('<div class="hero-title">About This Recommender</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">This application helps readers discover new books based on reading preferences and similar books.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### How the System Works")
        st.markdown(
            """
            * **Reader Community Patterns:** Understands which books readers frequently read and enjoy together, helping you discover books with similar appeal.
            * **Book Details & Author Style:** Analyzes book authors, titles, and publishers to find other books sharing similar writing and publishing characteristics.
            * **Balanced Suggestions:** Blends both reader community interest and author similarity to provide varied and relevant recommendations.
            * **Ready for New Readers:** Whether you have a long reading list or are discovering books for the first time, the system offers instant recommendations.
            """
        )

        st.markdown("---")
        st.markdown("### Catalog & Data")
        st.markdown(
            """
            The recommendation catalog includes hundreds of beloved books and thousands of ratings from avid readers in the Book-Crossing reading community.
            """
        )


if __name__ == "__main__":
    main()
