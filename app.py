
"""
NewsPulse — Flask Backend
Authentication : MySQL + bcrypt password hashing
Run            : python app.py
Open           : http://localhost:5000
"""

from dotenv import load_dotenv
load_dotenv()

import os
import io
import re
import base64
import shutil
import warnings
import requests
import bcrypt
import mysql.connector
import pandas as pd
import numpy as np

from collections import Counter
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
    redirect,
    url_for
)

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

app = Flask(__name__)
print("RUNNING FILE:", __file__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "newspulse_dev_secret_change_me"
)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==========================================================
# MYSQL CONFIG
# ==========================================================
print("DB_HOST =", os.environ.get("DB_HOST"))
print("DB_PORT =", os.environ.get("DB_PORT"))
print("DB_USER =", os.environ.get("DB_USER"))
print("DB_NAME =", os.environ.get("DB_NAME"))


DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ.get("DB_PORT") or 3306),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
}
# ==========================================================
# DATABASE HELPERS
# ==========================================================

def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():

    cfg = {k: v for k, v in DB_CONFIG.items() if k != "database"}

    conn = mysql.connector.connect(**cfg)
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`
        CHARACTER SET utf8mb4
        COLLATE utf8mb4_unicode_ci
        """
    )

    cur.execute(f"USE `{DB_CONFIG['database']}`")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            role ENUM('user','admin') DEFAULT 'user',
            theme ENUM('dark','light') DEFAULT 'dark',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            source VARCHAR(255),
            published VARCHAR(100),
            sentiment VARCHAR(20),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
            REFERENCES users(id)
            ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            query VARCHAR(255) NOT NULL,
            searched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
            REFERENCES users(id)
            ON DELETE CASCADE
        )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS upload_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        filename VARCHAR(255) NOT NULL,
        rows_count INT DEFAULT 0,
        columns_count INT DEFAULT 0,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
    )
""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS dataset_uploads (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        rows_count INT,
        columns_count INT,
        uploaded_by INT,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(uploaded_by)
        REFERENCES users(id)
        ON DELETE SET NULL
    )
""")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        id INT AUTO_INCREMENT PRIMARY KEY,

        title TEXT,
        description TEXT,

        source VARCHAR(255),

        published_date VARCHAR(100),

        sentiment_label VARCHAR(50),

        uploaded_by INT,

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(uploaded_by)
        REFERENCES users(id)
        ON DELETE SET NULL
    )
""")

    conn.commit()

    cur.execute(
        "SELECT id FROM users WHERE role='admin' LIMIT 1"
    )

    if not cur.fetchone():

        hashed = bcrypt.hashpw(
            b"admin123",
            bcrypt.gensalt()
        ).decode()

        cur.execute(
            """
            INSERT INTO users
            (username,email,password,role)
            VALUES (%s,%s,%s,%s)
            """,
            (
                "admin",
                "admin@newspulse.com",
                hashed,
                "admin"
            )
        )

        conn.commit()

        print(
            "Default admin created -> "
            "username: admin | password: admin123"
        )

    cur.close()
    conn.close()


init_db()


# ==========================================================
# AUTH DECORATORS
# ==========================================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if (
            not session.get("user_id")
            or session.get("role") != "admin"
        ):
            return jsonify({"error": "Unauthorized"}), 403

        return f(*args, **kwargs)

    return wrapper


# ==========================================================
# NLP / ML
# ==========================================================

POS_KEYWORDS = [
    'win','success','growth','develop','achiev',
    'innovat','launch','boost','improve',
    'benefit','profit','advance','excel',
    'strong','progress','celebrat','record',
    'best','great','top','award','invest',
    'rise','gain','increas','expand',
    'lead','new','upgrad','breakthrough',
    'renew','sustain','commit','promot',
    'support','partner','collabor',
    'inaugurat','open','approv','fund',
    'reform','opportun','thrive',
    'recover','posit','relief',
    'deal','partnership','tech',
    'ai','digital','smart'
]

NEG_KEYWORDS = [
    'war','conflict','attack','kill',
    'death','crisis','loss','fail',
    'declin','drop','fall','risk',
    'threat','violenc','disast',
    'protest','arrest','terror',
    'fraud','corrupt','accident',
    'critical','problem','controversi',
    'tension','strike','flood',
    'earthquake','recession',
    'unemploy','debt','breach',
    'hack','victim','collapse',
    'resign','shoot','murder',
    'disput','warn','sanction',
    'suspend','close','penalt'
]


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)

    return re.sub(r'\\s+', ' ', text).strip()


def compute_score(text):

    tl = text.lower()

    return (
        sum(1 for k in POS_KEYWORDS if k in tl)
        -
        sum(1 for k in NEG_KEYWORDS if k in tl)
    )


class RichKeywordFeatures(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        rows = []

        for text in X:

            tl = text.lower()

            pos = sum(
                1 for k in POS_KEYWORDS if k in tl
            )

            neg = sum(
                1 for k in NEG_KEYWORDS if k in tl
            )

            net = pos - neg

            rows.append([
                pos,
                neg,
                net,
                max(net - 1, 0),
                max(-1 - net, 0),
                float(pos) / max(neg, 1),
                float(net ** 2)
            ])

        return csr_matrix(
            np.array(rows, dtype=float)
        )


def fig_to_base64(fig):

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=110,
        facecolor=fig.get_facecolor()
    )

    buf.seek(0)

    image = base64.b64encode(
        buf.read()
    ).decode("utf-8")

    plt.close(fig)

    return image



# ==========================================================
# DATA LOADING
# ==========================================================

def load_data():

    csv_path = os.path.join(
        os.path.dirname(__file__),
        "gnews_data_cleaned.csv"
    )

    df = pd.read_csv(csv_path)

    # Fix missing columns
    if "Description" not in df.columns:
        df["Description"] = ""

    if "PublishedDate" in df.columns:
        df.rename(
            columns={
                "PublishedDate": "Published Date"
            },
            inplace=True
        )

    df = df.drop_duplicates()

    if "Title" in df.columns:
        df = df.dropna(subset=["Title"])

    df["full_text"] = (
        df["Title"].fillna("")
        + " " +
        df["Description"].fillna("")
    )

    df["cleaned_text"] = (
        df["full_text"]
        .apply(clean_text)
    )

    df["kw_score"] = (
        df["cleaned_text"]
        .apply(compute_score)
    )

    p25 = df["kw_score"].quantile(0.25)
    p60 = df["kw_score"].quantile(0.60)

    def label(score):

        if score >= p60:
            return "Positive"

        if score <= p25:
            return "Negative"

        return "Neutral"

    df["sentiment_label"] = (
        df["kw_score"]
        .apply(label)
    )

    if "Published Date" in df.columns:

        df["pub_date"] = pd.to_datetime(
            df["Published Date"],
            errors="coerce"
        )

    return df, p25, p60


# ==========================================================
# INITIAL DATASET LOAD
# ==========================================================

DF, P25, P60 = load_data()

TFIDF_VIZ = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
    max_df=0.95
)

TFIDF_MATRIX = TFIDF_VIZ.fit_transform(
    DF["cleaned_text"]
)

FEATURE_NAMES = (
    TFIDF_VIZ.get_feature_names_out()
)

MODEL_STORE = {}

# ==========================================================
# PAGE ROUTES
# ==========================================================

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/register")
def register_page():

    if session.get("user_id"):

        return redirect(
            url_for(
                "user_dashboard"
                if session.get("role") == "user"
                else "admin_dashboard"
            )
        )

    return render_template(
        "register.html"
    )


@app.route("/user/login")
def user_login_page():

    if (
        session.get("user_id")
        and session.get("role") == "user"
    ):
        return redirect(
            url_for("user_dashboard")
        )

    return render_template(
        "user_login.html"
    )


@app.route("/user/dashboard")
def user_dashboard():

    if not session.get("user_id"):
        return redirect(
            url_for("user_login_page")
        )

    return render_template(
        "index.html",
        portal="user"
    )


@app.route("/admin/login")
def admin_login_page():

    if (
        session.get("user_id")
        and session.get("role") == "admin"
    ):
        return redirect(
            url_for("admin_dashboard")
        )

    return render_template(
        "admin_login.html"
    )


@app.route("/admin/dashboard")
def admin_dashboard():

    if (
        not session.get("user_id")
        or session.get("role") != "admin"
    ):
        return redirect(
            url_for("admin_login_page")
        )

    return render_template(
        "index.html",
        portal="admin"
    )


# ==========================================================
# AUTH APIs
# ==========================================================

@app.route("/api/register", methods=["POST"])
def register():

    data = request.json or {}

    username = data.get(
        "username",
        ""
    ).strip()

    email = data.get(
        "email",
        ""
    ).strip().lower()

    password = data.get(
        "password",
        ""
    )

    if not username or not email or not password:
        return jsonify({
            "success": False,
            "error": "All fields are required."
        }), 400

    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    try:

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO users
            (username,email,password,role)
            VALUES (%s,%s,%s,'user')
            """,
            (
                username,
                email,
                hashed
            )
        )

        conn.commit()

        cur.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Account created successfully"
        })

    except mysql.connector.IntegrityError:

        return jsonify({
            "success": False,
            "error": "Username or Email already exists"
        }), 409


@app.route("/api/user/login", methods=["POST"])
def user_login_api():

    data = request.json or {}

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT *
        FROM users
        WHERE username=%s
        AND role='user'
        """,
        (username,)
    )

    user = cur.fetchone()

    cur.close()
    conn.close()

    if (
        not user
        or not bcrypt.checkpw(
            password.encode(),
            user["password"].encode()
        )
    ):
        return jsonify({
            "success": False,
            "error": "Invalid credentials"
        }), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["theme"] = user["theme"]

    return jsonify({
        "success": True,
        "redirect": "/user/dashboard"
    })


@app.route("/api/admin/login", methods=["POST"])
def admin_login_api():

    data = request.json or {}

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT *
        FROM users
        WHERE username=%s
        AND role='admin'
        """,
        (username,)
    )

    user = cur.fetchone()

    cur.close()
    conn.close()

    if (
        not user
        or not bcrypt.checkpw(
            password.encode(),
            user["password"].encode()
        )
    ):
        return jsonify({
            "success": False,
            "error": "Invalid admin credentials"
        }), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = "admin"
    session["theme"] = user["theme"]

    return jsonify({
        "success": True,
        "redirect": "/admin/dashboard"
    })


@app.route("/api/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "success": True
    })


@app.route("/api/session")
def get_session():

    return jsonify({
        "user_id": session.get("user_id"),
        "username": session.get("username", ""),
        "role": session.get("role", ""),
        "theme": session.get("theme", "dark"),
        "is_user": session.get("role") == "user",
        "is_admin": session.get("role") == "admin"
    })


# ==========================================================
# USER DASHBOARD APIs
# ==========================================================

@app.route("/api/stats")
@login_required
def stats():

    sc = DF["sentiment_label"].value_counts()

    return jsonify({
        "total": int(len(DF)),
        "positive": int(sc.get("Positive", 0)),
        "negative": int(sc.get("Negative", 0)),
        "neutral": int(sc.get("Neutral", 0)),
        "sources": int(
            DF["Source"].nunique()
        ) if "Source" in DF.columns else 0
    })


@app.route("/api/articles")
@login_required
def articles():

    sentiment = request.args.get(
        "sentiment",
        "All"
    )

    source = request.args.get(
        "source",
        "All"
    )

    search = request.args.get(
        "search",
        ""
    ).strip()

    page = int(
        request.args.get("page", 1)
    )

    per_page = int(
        request.args.get("per_page", 20)
    )

    df = DF.copy()

    if sentiment != "All":
        df = df[
            df["sentiment_label"] == sentiment
        ]

    if (
        source != "All"
        and "Source" in df.columns
    ):
        df = df[
            df["Source"] == source
        ]

    if search:

        mask = (
            df["Title"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            df["Description"].str.contains(
                search,
                case=False,
                na=False
            )
        )

        df = df[mask]

    total = len(df)

    start = (
        (page - 1) * per_page
    )

    subset = df.iloc[
        start:start + per_page
    ]

    cols = [
        c for c in [
            "Title",
            "Description",
            "Source",
            "Published Date",
            "sentiment_label"
        ]
        if c in subset.columns
    ]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "articles":
        subset[cols]
        .fillna("")
        .to_dict(
            orient="records"
        )
    })


@app.route("/api/sources")
@login_required
def sources():

    if "Source" not in DF.columns:
        return jsonify([])

    return jsonify(
        sorted(
            DF["Source"]
            .dropna()
            .unique()
            .tolist()
        )
    )


@app.route("/api/trending")
@login_required
def trending():

    freq = Counter(
        " ".join(
            DF["cleaned_text"]
        ).split()
    )

    return jsonify([
        {
            "word": w,
            "count": c
        }
        for w, c in freq.most_common(20)
    ])


@app.route("/api/sentiment_chart")
@login_required
def sentiment_chart():

    sc = DF[
        "sentiment_label"
    ].value_counts()

    return jsonify({
        "labels":
        sc.index.tolist(),

        "values":
        sc.values.tolist()
    })


@app.route("/api/sentiment_trend")
@login_required
def sentiment_trend():

    if "pub_date" not in DF.columns:

        return jsonify({
            "labels": [],
            "positive": [],
            "negative": [],
            "neutral": []
        })

    df = DF.dropna(
        subset=["pub_date"]
    ).copy()

    df["month"] = (
        df["pub_date"]
        .dt.to_period("M")
        .astype(str)
    )

    grp = (
        df.groupby(
            ["month", "sentiment_label"]
        )
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    labels = grp.index.tolist()

    return jsonify({
        "labels": labels,

        "positive": [
            int(
                grp.loc[m, "Positive"]
            )
            if "Positive" in grp.columns
            else 0
            for m in labels
        ],

        "negative": [
            int(
                grp.loc[m, "Negative"]
            )
            if "Negative" in grp.columns
            else 0
            for m in labels
        ],

        "neutral": [
            int(
                grp.loc[m, "Neutral"]
            )
            if "Neutral" in grp.columns
            else 0
            for m in labels
        ]
    })

# ==========================================================
# BOOKMARKS API
# ==========================================================

@app.route("/api/bookmarks")
@login_required
def get_bookmarks():

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        "SELECT * FROM bookmarks WHERE user_id=%s ORDER BY created_at DESC",
        (session["user_id"],)
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    for row in rows:
        row["created_at"] = str(row["created_at"])

    return jsonify(rows)


@app.route("/api/bookmarks/add", methods=["POST"])
@login_required
def add_bookmark():

    data = request.json or {}

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO bookmarks
        (user_id,title,description,source,published,sentiment)
        VALUES (%s,%s,%s,%s,%s,%s)
        """,
        (
            session["user_id"],
            data.get("title", ""),
            data.get("description", ""),
            data.get("source", ""),
            data.get("published", ""),
            data.get("sentiment", "")
        )
    )

    conn.commit()

    cur.close()
    conn.close()

    return jsonify({
        "success": True
    })


@app.route("/api/bookmarks/remove", methods=["POST"])
@login_required
def remove_bookmark():

    title = (request.json or {}).get("title", "")

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM bookmarks WHERE user_id=%s AND title=%s",
        (session["user_id"], title)
    )

    conn.commit()

    cur.close()
    conn.close()

    return jsonify({
        "success": True
    })

# ==========================================================
# LIVE NEWS API
# ==========================================================

@app.route("/api/live-news")
@login_required
def live_news():

    query = request.args.get(
        "query",
        ""
    ).strip()

    if not query:
        return jsonify([])

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params
        )

        data = response.json()

        articles = []

        for article in data.get(
            "articles",
            []
        ):

            articles.append({
                "title": article.get(
                    "title",
                    ""
                ),
                "description": article.get(
                    "description",
                    ""
                ),
                "source": article.get(
                    "source",
                    {}
                ).get(
                    "name",
                    ""
                ),
                "published": article.get(
                    "publishedAt",
                    ""
                ),
                "url": article.get(
                    "url",
                    ""
                )
            })

        return jsonify(articles)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
    



# ==========================================================
# UPLOAD DATASET API
# ==========================================================
@app.route("/api/upload-file", methods=["POST"])
@login_required
def upload_file():

    global DF
    global P25
    global P60
    global TFIDF_VIZ
    global TFIDF_MATRIX
    global FEATURE_NAMES

    if "file" not in request.files:
        return jsonify({
            "message": "No file selected"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "message": "No file selected"
        }), 400

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    try:

        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(filepath)

        else:
            return jsonify({
                "message": "Only CSV or Excel allowed"
            }), 400

        shutil.copy(
            filepath,
            "gnews_data_cleaned.csv"
        )

        DF, P25, P60 = load_data()

        TFIDF_VIZ = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )

        TFIDF_MATRIX = TFIDF_VIZ.fit_transform(
            DF["cleaned_text"]
        )

        FEATURE_NAMES = (
            TFIDF_VIZ.get_feature_names_out()
        )

        conn = get_db()
        cur = conn.cursor()
        

        # Store dataset rows in MySQL
        cur.execute("DELETE FROM news_articles")
        conn.commit()
        
        for _, row in df.iterrows():

            cur.execute("""
                INSERT INTO news_articles
                (
                    title,
                    description,
                    source,
                    published_date,
                    sentiment_label,
                    uploaded_by
                )
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (

                str(row.get("Title", "")),
                str(row.get("Description", "")),
                str(row.get("Source", "")),
                str(row.get("Published Date", "")),
                str(row.get("sentiment_label", "")),
                session["user_id"]

            ))

        # Upload history
        cur.execute("""
            INSERT INTO upload_history
            (
                user_id,
                filename,
                rows_count,
                columns_count
            )
            VALUES (%s,%s,%s,%s)
        """, (
            session["user_id"],
            filename,
            len(df),
            len(df.columns)
        ))

        conn.commit()

        cur.close()
        conn.close()

        print("DATASET RELOADED")
        print("ROWS:", len(DF))

        return jsonify({
            "message": "File uploaded successfully",
            "rows": len(df),
            "columns": len(df.columns),
            "filename": filename,
            "preview": df.head(10)
                        .fillna("")
                        .to_dict(orient="records")
        })

    except Exception as e:

        return jsonify({
            "message": str(e)
        }), 500

@app.route("/api/search_history")
@login_required
def get_search_history():

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            id,
            query,
            searched_at
        FROM search_history
        WHERE user_id=%s
        ORDER BY searched_at DESC
        LIMIT 20
    """, (session["user_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    for row in rows:
        row["searched_at"] = str(row["searched_at"])

    return jsonify(rows)

@app.route("/api/profile")
@login_required
def get_profile():

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            id,
            username,
            email,
            role,
            theme,
            created_at
        FROM users
        WHERE id=%s
    """, (session["user_id"],))

    user = cur.fetchone()

    if not user:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    cur.execute("""
        SELECT COUNT(*)
        FROM bookmarks
        WHERE user_id=%s
    """, (session["user_id"],))

    bookmark_count = cur.fetchone()["COUNT(*)"]

    user["bookmark_count"] = bookmark_count
    user["created_at"] = str(user["created_at"])

    cur.close()
    conn.close()

    return jsonify(user)

@app.route("/api/profile/update", methods=["POST"])
@login_required
def update_profile():

    data = request.json

    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    conn = get_db()
    cur = conn.cursor()

    if email:

        cur.execute("""
            UPDATE users
            SET email=%s
            WHERE id=%s
        """, (
            email,
            session["user_id"]
        ))

    if password:

        hashed = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        ).decode()

        cur.execute("""
            UPDATE users
            SET password=%s
            WHERE id=%s
        """, (
            hashed,
            session["user_id"]
        ))

    conn.commit()

    cur.close()
    conn.close()

    return jsonify({
        "message": "Profile updated successfully"
    })


@app.route("/api/upload_history")
@login_required
def upload_history():

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            filename,
            rows_count,
            columns_count,
            uploaded_at
        FROM upload_history
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
    """, (session["user_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(rows)

@app.route("/api/dataset-stats")
@login_required
def dataset_stats():

    return jsonify({

        "rows": len(DF),

        "columns": len(DF.columns),

        "missing_values": int(
            DF.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            DF.duplicated().sum()
        ),

        "column_names": list(
            DF.columns
        ),

        "data_types": {
            col: str(dtype)
            for col, dtype in DF.dtypes.items()
        }
    })

@app.route("/api/dataset-preview")
@login_required
def dataset_preview():

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            title,
            source,
            sentiment_label
        FROM news_articles
        LIMIT 50
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(rows)


if __name__ == "__main__":

    print(f"RUNNING FILE: {__file__}")
    print("\n🚀 NewsPulse → http://localhost:5000\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )