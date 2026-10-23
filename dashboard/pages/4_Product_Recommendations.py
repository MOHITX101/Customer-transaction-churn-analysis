from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT / "models/recommendations/2020-01-31"
)

FILES = {
    "Article relationships": MODEL_DIR / "item_neighbors.parquet",
    "Customer history": MODEL_DIR / "customer_article_history.parquet",
    "Popular products": MODEL_DIR / "popularity_ranking.parquet",
    "Product details": PROJECT_ROOT / "data/processed/articles.parquet",
}

TOP_K = 10
RESULT_KEY = "product_recommendations_result"


# =========================================================
# DATA ACCESS
# =========================================================
@st.cache_data(max_entries=4)
def load_example_customers(history_path, modified):
    with duckdb.connect() as connection:
        return connection.execute(
            """
            SELECT DISTINCT customer_id
            FROM read_parquet(?)
            WHERE customer_id IS NOT NULL
            ORDER BY customer_id
            LIMIT 20
            """,
            [history_path],
        ).df()["customer_id"].tolist()


@st.cache_data(max_entries=100)
def generate_recommendations(
    customer_id,
    history_path,
    neighbors_path,
    popularity_path,
    articles_path,
    file_versions,
):
    with duckdb.connect() as connection:
        # Register the saved components in this connection.
        connection.from_parquet(history_path).create_view("history")
        connection.from_parquet(neighbors_path).create_view("neighbors")
        connection.from_parquet(popularity_path).create_view("popularity")
        connection.from_parquet(articles_path).create_view("articles")

        history_summary = connection.execute(
            """
            SELECT
                COUNT(DISTINCT article_id) AS recent_articles,
                SUM(purchase_records) AS recent_purchase_records,
                MAX(last_purchase_date) AS latest_purchase
            FROM history
            WHERE customer_id = ?
            """,
            [customer_id],
        ).df()

        recommendations = connection.execute(
            """
            WITH customer_articles AS (
                SELECT DISTINCT article_id
                FROM history
                WHERE customer_id = ?
            ),

            candidate_scores AS (
                SELECT
                    n.neighbor_article_id AS article_id,
                    SUM(n.similarity_score) AS recommendation_score
                FROM customer_articles h
                INNER JOIN neighbors n
                    ON h.article_id = n.source_article_id
                GROUP BY n.neighbor_article_id
            ),

            ranked_personalized AS (
                SELECT
                    article_id,
                    recommendation_score,
                    ROW_NUMBER() OVER (
                        ORDER BY recommendation_score DESC, article_id
                    ) AS source_rank
                FROM candidate_scores
            ),

            personalized AS (
                SELECT
                    article_id,
                    recommendation_score,
                    source_rank,
                    0 AS source_priority,
                    'Personalized' AS recommendation_source
                FROM ranked_personalized
                WHERE source_rank <= 10
            ),

            fallback AS (
                SELECT
                    p.article_id,
                    0.0 AS recommendation_score,
                    p.recommendation_rank AS source_rank,
                    1 AS source_priority,
                    'Popular fallback' AS recommendation_source
                FROM popularity p
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM personalized r
                    WHERE r.article_id = p.article_id
                )
            ),

            combined AS (
                SELECT * FROM personalized
                UNION ALL
                SELECT * FROM fallback
            ),

            top_products AS (
                SELECT *
                FROM combined
                ORDER BY source_priority, source_rank, article_id
                LIMIT 10
            )

            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY source_priority, source_rank, article_id
                ) AS recommendation_rank,
                article_id,
                recommendation_source,
                recommendation_score
            FROM top_products
            ORDER BY recommendation_rank
            """,
            [customer_id],
        ).df()

        if recommendations.empty:
            raise ValueError("No recommendations could be generated.")

        if recommendations["article_id"].duplicated().any():
            raise ValueError("Duplicate recommended products were found.")

        if len(recommendations) != TOP_K:
            raise ValueError(
                f"Expected {TOP_K} products; found {len(recommendations)}."
            )

        # Join product descriptions after selecting only 10 products.
        connection.register("selected_recommendations", recommendations)

        detailed = connection.execute(
            """
            SELECT
                r.recommendation_rank,
                r.article_id,
                COALESCE(a.prod_name, 'Product name unavailable')
                    AS product_name,
                COALESCE(a.product_group_name, 'Unknown')
                    AS product_group,
                COALESCE(a.product_type_name, 'Unknown')
                    AS product_type,
                COALESCE(a.colour_group_name, 'Unknown')
                    AS colour,
                r.recommendation_source,
                r.recommendation_score
            FROM selected_recommendations r
            LEFT JOIN articles a
                ON CAST(r.article_id AS VARCHAR)
                   = CAST(a.article_id AS VARCHAR)
            ORDER BY r.recommendation_rank
            """
        ).df()

        if len(detailed) != TOP_K:
            raise ValueError(
                "The product-details join changed the recommendation count. "
                "Check article ID uniqueness."
            )

    return detailed, history_summary


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Product Recommendations")

st.write(
    "Generate 10 product suggestions using a customer's recent "
    "purchase history and the saved item-to-item recommendation model."
)

st.info(
    "Historical snapshot: 31 January 2020. "
    "The model was evaluated on purchases from 1–7 February 2020."
)


# =========================================================
# CHECK FILES
# =========================================================
missing = [
    name for name, path in FILES.items()
    if not path.is_file()
]

with st.expander("Model and data availability", expanded=bool(missing)):
    st.dataframe(
        pd.DataFrame([
            {
                "Component": name,
                "File": path.name,
                "Status": "Available" if path.is_file() else "Missing",
            }
            for name, path in FILES.items()
        ]),
        hide_index=True,
        use_container_width=True,
    )

if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

file_versions = tuple(
    (str(path), path.stat().st_mtime_ns)
    for path in FILES.values()
)

previous_result = st.session_state.get(RESULT_KEY)

if (
    previous_result is not None
    and previous_result.get("file_versions") != file_versions
):
    st.session_state.pop(RESULT_KEY, None)

try:
    example_customers = load_example_customers(
        str(FILES["Customer history"]),
        FILES["Customer history"].stat().st_mtime_ns,
    )
except Exception as error:
    st.error(f"Could not load customer examples: {error}")
    st.stop()

if not example_customers:
    st.error("No customers were found in the saved purchase history.")
    st.stop()


# =========================================================
# CUSTOMER SELECTION
# =========================================================
st.subheader("Choose a customer")

st.caption(
    "Examples come from the recommendation model's recent-history file. "
    "You can also paste a full customer ID."
)

with st.form("recommendation_form"):
    example_id = st.selectbox(
        "Example customer",
        options=example_customers,
    )

    entered_id = st.text_input(
        "Or paste a full customer ID",
        placeholder="Leave blank to use the selected example",
    )

    submitted = st.form_submit_button(
        "Generate recommendations",
        type="primary",
    )

if submitted:
    selected_id = entered_id.strip() or example_id
    st.session_state.pop(RESULT_KEY, None)

    try:
        with st.spinner("Finding recommended products..."):
            products, history_summary = generate_recommendations(
                selected_id,
                str(FILES["Customer history"]),
                str(FILES["Article relationships"]),
                str(FILES["Popular products"]),
                str(FILES["Product details"]),
                file_versions,
            )

            st.session_state[RESULT_KEY] = {
                "customer_id": selected_id,
                "products": products,
                "history_summary": history_summary,
                "file_versions": file_versions,
            }

    except Exception as error:
        st.error(f"Could not generate recommendations: {error}")


# =========================================================
# RESULTS
# =========================================================
result = st.session_state.get(RESULT_KEY)

if result is None:
    st.caption(
        "Choose a customer and click Generate recommendations."
    )

else:
    products = result["products"]
    history_summary = result["history_summary"].iloc[0]

    personalized_count = int(
        (products["recommendation_source"] == "Personalized").sum()
    )

    fallback_count = len(products) - personalized_count
    recent_articles = int(history_summary["recent_articles"])

    st.divider()
    st.subheader("Recommended products")

    st.caption("Customer ID")
    st.code(result["customer_id"], language=None)

    col1, col2, col3 = st.columns(3)

    col1.metric("Recommended products", len(products))
    col2.metric("Personalized suggestions", personalized_count)
    col3.metric("Popular fallback suggestions", fallback_count)

    if recent_articles == 0:
        st.info(
            "This ID has no history in the model's saved 90-day window. "
            "The list therefore contains popular products. "
            "This can also happen if the ID was entered incorrectly."
        )
    elif personalized_count == 0:
        st.info(
            "Recent purchase history exists, but it did not produce "
            "eligible product matches. Popular products fill the list."
        )
    elif fallback_count > 0:
        st.info(
            "Personalized matches appear first. Popular products fill "
            "the remaining places."
        )

    st.caption(
        f"Distinct products in this customer's saved recent history: "
        f"{recent_articles:,}"
    )

    # Two product cards per row.
    for start in range(0, len(products), 2):
        columns = st.columns(2)

        for column, (_, product) in zip(
            columns,
            products.iloc[start:start + 2].iterrows(),
        ):
            with column:
                with st.container(border=True):
                    st.caption(
                        f"RECOMMENDATION "
                        f"{int(product['recommendation_rank']):02d}"
                    )

                    st.subheader(str(product["product_name"]))

                    st.write(
                        f"**Category:** {product['product_group']}"
                    )
                    st.write(
                        f"**Product type:** {product['product_type']}"
                    )
                    st.write(f"**Colour:** {product['colour']}")

                    st.caption(
                        f"Article ID: {product['article_id']}"
                    )

                    st.write(
                        f"**Source:** "
                        f"{product['recommendation_source']}"
                    )

    with st.expander("How these recommendations are generated"):
        st.write(
            "The saved model links products purchased by overlapping "
            "sets of customers. For this customer, it adds the saved "
            "similarity scores across products in their recent history "
            "and ranks the resulting candidates."
        )

        st.write(
            "Popular products fill any unused positions. "
            "Previously purchased products remain eligible, matching "
            "the evaluation setup."
        )

        st.caption(
            "Recommendation scores rank products; they are not purchase "
            "probabilities. The data does not establish current stock "
            "availability."
        )

    with st.expander("View recommendation details"):
        st.dataframe(
            products,
            hide_index=True,
            use_container_width=True,
        )

    export = products.copy()
    export.insert(0, "customer_id", result["customer_id"])
    export.insert(1, "recommendation_as_of_date", "2020-01-31")

    st.download_button(
        label="Download recommendations",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="customer_recommendations_2020-01-31.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Academic recommendation prototype using historical H&M data."
)