import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DevSalary AI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_salary_model.joblib"
TRAIN_PATH = "data/train.csv"

# Your trained model predicts compensation in USD.
# This value converts the prediction to INR for display.
USD_TO_INR = 90.0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(TRAIN_PATH)


model = load_model()
df = load_data()


# ============================================================
# DATASET OPTIONS
# ============================================================

countries = sorted(
    df["Country"].dropna().unique().tolist()
)

developer_types = sorted(
    df["DevType"].dropna().unique().tolist()
)

education_levels = sorted(
    df["EdLevel"].dropna().unique().tolist()
)

employment_types = sorted(
    df["Employment"].dropna().unique().tolist()
)

organization_sizes = sorted(
    df["OrgSize"].dropna().unique().tolist()
)

remote_options = sorted(
    df["RemoteWork"].dropna().unique().tolist()
)

role_options = sorted(
    df["ICorPM"].dropna().unique().tolist()
)

industries = sorted(
    df["Industry"].dropna().unique().tolist()
)


# ============================================================
# PROGRAMMING LANGUAGES
# ============================================================

languages = {
    "JavaScript": "lang_javascript",
    "HTML / CSS": "lang_html_css",
    "SQL": "lang_sql",
    "Python": "lang_python",
    "Bash / Shell": "lang_bash_shell",
    "TypeScript": "lang_typescript",
    "C#": "lang_csharp",
    "Java": "lang_java",
    "PowerShell": "lang_powershell",
    "C++": "lang_cpp",
    "PHP": "lang_php",
    "C": "lang_c",
    "Go": "lang_go",
    "Rust": "lang_rust",
    "Kotlin": "lang_kotlin"
}


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_inr" not in st.session_state:
    st.session_state.prediction_inr = None

if "prediction_usd" not in st.session_state:
    st.session_state.prediction_usd = None


# ============================================================
# PREMIUM DARK UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(108, 95, 225, 0.08),
                transparent 28%
            ),
            #080a10;

        color: #f4f5f8;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 32px;
        padding-bottom: 60px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ======================================================
       BRAND
       ====================================================== */

    .brand {
        font-size: 25px;
        font-weight: 760;
        letter-spacing: -1.1px;
        color: #f5f6fa;
        line-height: 1;
    }

    .brand-ai {
        color: #8077f5;
    }

    .brand-tagline {
        color: #626a7b;
        font-size: 10px;
        font-weight: 650;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 7px;
    }

    .brand-line {
        height: 1px;
        background: #202430;
        margin-top: 21px;
        margin-bottom: 55px;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero-label {
        color: #767e91;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .hero-description {
        color: #858d9f;
        font-size: 15px;
        line-height: 1.7;
        max-width: 680px;
        margin-top: 17px;
        margin-bottom: 38px;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {
        color: #f5f6fa !important;
        font-weight: 760 !important;
        font-size: 50px !important;
        letter-spacing: -2.7px !important;
        line-height: 1.05 !important;
    }

    h2 {
        color: #f0f1f5 !important;
        font-weight: 680 !important;
        letter-spacing: -0.7px !important;
    }

    h3 {
        color: #e8eaf0 !important;
        font-weight: 650 !important;
    }

    p {
        color: #858d9f;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: #737b8e !important;
        font-size: 13px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #aaa3ff !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #756cf2 !important;
    }


    /* ======================================================
       INPUT LABELS
       ====================================================== */

    div[data-testid="stWidgetLabel"] p {
        color: #9fa6b6 !important;
        font-size: 12px !important;
        font-weight: 550 !important;
    }


    /* ======================================================
       INPUTS
       ====================================================== */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #11141c !important;
        border: 1px solid #292e3b !important;
        border-radius: 8px !important;
        min-height: 43px;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #665de5 !important;
        box-shadow: 0 0 0 1px rgba(102, 93, 229, 0.13) !important;
    }

    div[data-baseweb="input"] input {
        color: #e9ebf1 !important;
    }

    div[data-baseweb="select"] span {
        color: #e0e3ea !important;
    }


    /* ======================================================
       CHECKBOXES
       ====================================================== */

    [data-testid="stCheckbox"] label {
        color: #a1a8b7 !important;
        font-size: 12px !important;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        width: 100%;
        min-height: 49px;

        background: #7067ef !important;
        color: #ffffff !important;

        border: 1px solid #8179ff !important;
        border-radius: 8px !important;

        font-size: 13px !important;
        font-weight: 700 !important;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #7b72f7 !important;
        border-color: #918aff !important;
        transform: translateY(-1px);
    }


    /* ======================================================
       METRICS
       ====================================================== */

    [data-testid="stMetric"] {
        background: #10141d !important;
        border: 1px solid #282d3a !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #777f91 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f3f7 !important;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        border-color: #202430 !important;
        margin: 28px 0 !important;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-text {
        color: #555d6e;
        font-size: 10px;
        text-align: center;
        letter-spacing: 0.4px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BRAND HEADER
# ============================================================

st.markdown(
    """
    <div class="brand">
        DevSalary <span class="brand-ai">AI</span>
    </div>

    <div class="brand-tagline">
        Developer salary intelligence
    </div>

    <div class="brand-line"></div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-label">Salary estimator</div>',
    unsafe_allow_html=True
)

st.title(
    "What could your experience be worth?"
)

st.markdown(
    """
    <div class="hero-description">
        Estimate your annual developer compensation based on
        experience, education, location, professional role,
        organization and technical skills.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

predict_tab, about_tab = st.tabs(
    [
        "Predict Salary",
        "About"
    ]
)


# ============================================================
# PREDICT SALARY
# ============================================================

with predict_tab:

    left, right = st.columns(
        [1.65, 1],
        gap="large"
    )


    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left:

        st.subheader(
            "Professional profile"
        )

        st.caption(
            "Tell us about your experience and current career profile."
        )

        st.markdown("---")


        # ====================================================
        # EXPERIENCE
        # ====================================================

        st.markdown(
            "#### Experience"
        )

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:

            work_exp = st.number_input(
                "Years of work experience",
                min_value=0.0,
                max_value=50.0,
                value=4.0,
                step=0.5
            )

        with exp_col2:

            years_code = st.number_input(
                "Years of coding experience",
                min_value=0.0,
                max_value=60.0,
                value=4.0,
                step=0.5
            )


        st.markdown("")


        # ====================================================
        # PROFESSIONAL INFORMATION
        # ====================================================

        st.markdown(
            "#### Professional information"
        )

        col1, col2 = st.columns(2)


        with col1:

            country = st.selectbox(
                "Country",
                countries,
                index=(
                    countries.index("India")
                    if "India" in countries
                    else 0
                )
            )

            education = st.selectbox(
                "Education level",
                education_levels,
                index=(
                    education_levels.index(
                        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)"
                    )
                    if
                    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)"
                    in education_levels
                    else 0
                )
            )

            organization = st.selectbox(
                "Organization size",
                organization_sizes
            )

            industry = st.selectbox(
                "Industry",
                industries,
                index=(
                    industries.index(
                        "Software Development"
                    )
                    if "Software Development"
                    in industries
                    else 0
                )
            )


        with col2:

            developer_type = st.selectbox(
                "Developer type",
                developer_types,
                index=(
                    developer_types.index(
                        "Developer, back-end"
                    )
                    if "Developer, back-end"
                    in developer_types
                    else 0
                )
            )

            employment = st.selectbox(
                "Employment type",
                employment_types
            )

            remote_work = st.selectbox(
                "Remote work",
                remote_options
            )

            role = st.selectbox(
                "Role",
                role_options
            )


        st.markdown("")


        # ====================================================
        # TECHNICAL SKILLS
        # ====================================================

        st.markdown(
            "#### Technical skills"
        )

        st.caption(
            "Select the technologies you currently work with."
        )

        tech_cols = st.columns(5)

        selected_languages = {}


        for index, (
            display_name,
            column_name
        ) in enumerate(languages.items()):

            with tech_cols[index % 5]:

                selected_languages[column_name] = (
                    st.checkbox(
                        display_name,
                        value=False,
                        key=f"language_{column_name}"
                    )
                )


        selected_count = sum(
            selected_languages.values()
        )


        st.caption(
            f"{selected_count} technology"
            + (
                " selected"
                if selected_count == 1
                else "s selected"
            )
        )


        st.markdown("")


        # ====================================================
        # PREDICT BUTTON
        # ====================================================

        predict_button = st.button(
            "Estimate My Salary  →",
            use_container_width=True
        )


    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right:

        st.subheader(
            "Estimated salary"
        )

        st.caption(
            "Your estimated annual developer compensation."
        )

        st.markdown("---")


        # ====================================================
        # BEFORE PREDICTION
        # ====================================================

        if st.session_state.prediction_inr is None:

            st.metric(
                "ANNUAL COMPENSATION",
                "₹ —"
            )

            st.caption(
                "Complete your profile to generate your estimate."
            )


        # ====================================================
        # AFTER PREDICTION
        # ====================================================

        else:

            annual_inr = (
                st.session_state.prediction_inr
            )

            monthly_inr = (
                annual_inr / 12
            )

            lpa = (
                annual_inr / 100000
            )


            st.metric(
                "ESTIMATED ANNUAL COMPENSATION",
                f"₹{annual_inr:,.0f}"
            )


            st.caption(
                f"Approximately ₹{lpa:.1f} LPA"
            )


            st.metric(
                "ESTIMATED MONTHLY",
                f"₹{monthly_inr:,.0f}"
            )


            st.caption(
                "Estimate based on your selected profile."
            )


        # ====================================================
        # PROFILE SUMMARY
        # ====================================================

        st.markdown("---")

        st.markdown(
            "#### Profile summary"
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.caption(
                "Experience"
            )

            st.write(
                f"{work_exp:g} years"
            )


            st.caption(
                "Country"
            )

            st.write(
                country
            )


            st.caption(
                "Developer"
            )

            st.write(
                developer_type
            )


        with summary_col2:

            st.caption(
                "Coding experience"
            )

            st.write(
                f"{years_code:g} years"
            )


            st.caption(
                "Work mode"
            )

            st.write(
                remote_work
            )


            st.caption(
                "Technologies"
            )

            st.write(
                f"{selected_count} selected"
            )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        input_data = pd.DataFrame(
            {
                "WorkExp": [work_exp],
                "YearsCode": [years_code],
                "Country": [country],
                "DevType": [developer_type],
                "EdLevel": [education],
                "Employment": [employment],
                "OrgSize": [organization],
                "RemoteWork": [remote_work],
                "ICorPM": [role],
                "Industry": [industry]
            }
        )


        # ====================================================
        # LANGUAGE FLAGS
        # ====================================================

        for column_name in languages.values():

            input_data[column_name] = [
                int(
                    selected_languages[column_name]
                )
            ]


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_usd = float(
            model.predict(input_data)[0]
        )


        # ====================================================
        # INR CONVERSION
        # ====================================================

        prediction_inr = (
            prediction_usd * USD_TO_INR
        )


        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        st.session_state.prediction_usd = (
            prediction_usd
        )

        st.session_state.prediction_inr = (
            prediction_inr
        )


        st.rerun()


# ============================================================
# ABOUT
# ============================================================

with about_tab:

    st.subheader(
        "About DevSalary AI"
    )

    st.caption(
        "A machine learning application for estimating "
        "developer compensation."
    )

    st.markdown("---")


    about_col1, about_col2 = st.columns(2)


    with about_col1:

        st.markdown(
            "#### The project"
        )

        st.write(
            "DevSalary AI estimates developer compensation "
            "using professional experience, education, "
            "location, developer role, organization and "
            "technical skills."
        )

        st.write(
            "The application uses a Random Forest regression "
            "model trained on developer survey data."
        )


    with about_col2:

        st.markdown(
            "#### Technology"
        )

        st.write(
            "Python · Pandas · NumPy · Scikit-learn · "
            "Random Forest · Streamlit"
        )

        st.write(
            "The project includes data cleaning, "
            "preprocessing, exploratory analysis, "
            "model training, evaluation and prediction."
        )


    st.markdown("---")


    stat1, stat2, stat3, stat4 = st.columns(4)


    with stat1:

        st.metric(
            "CLEANED RECORDS",
            "21,806"
        )


    with stat2:

        st.metric(
            "INPUT FEATURES",
            "25"
        )


    with stat3:

        st.metric(
            "TEST RECORDS",
            "4,362"
        )


    with stat4:

        st.metric(
            "BEST MODEL",
            "Random Forest"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">
        DevSalary AI · Developer Salary Intelligence
    </div>
    """,
    unsafe_allow_html=True
)