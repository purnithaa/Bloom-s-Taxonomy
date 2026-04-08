"""
=============================================================
  Bloom's Taxonomy Question Classifier — Streamlit App
  Categories: K1 (Remember), K2 (Understand), K3 (Apply),
              K4 (Analyze), K5 (Evaluate), K6 (Create),
              NOT_BLOOMS
=============================================================
"""

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Bloom's Taxonomy Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #05050f;
  --surf:     #0c0c1d;
  --surf2:    #111126;
  --surf3:    #16162f;
  --border:   #1f1f3d;
  --border2:  #2d2d52;
  --text:     #ece9ff;
  --text2:    #b8b4d8;
  --muted:    #6e6a90;
  --violet:   #8b6fc7;
  --violet2:  #a98ee8;
  --pink:     #e066a8;
  --cyan:     #38d9c0;
  --gold:     #f5c842;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text); }

.stApp { background: var(--bg) !important; }
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 5%,  rgba(139,111,199,0.20) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 85% 75%, rgba(56,217,192,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 50% 55% at 55% 25%, rgba(224,102,168,0.10) 0%, transparent 55%);
  pointer-events: none; z-index: 0;
  animation: bgPulse 10s ease-in-out infinite alternate;
}
@keyframes bgPulse { from { opacity:1; } to { opacity:0.65; } }
.block-container { position:relative; z-index:1; padding-top:0 !important; padding-bottom:4rem !important; max-width:1260px !important; }
#MainMenu, footer, header { visibility:hidden; }
::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:var(--bg); } ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:99px; }

[data-testid="stSidebar"] { background:var(--surf) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding:1.5rem 1.1rem !important; }

.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; gap:0 !important; padding:0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:none !important; color:var(--muted) !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.9rem !important; font-weight:600 !important; padding:0.85rem 1.6rem !important; border-bottom:2px solid transparent !important; margin-bottom:-1px !important; transition:all 0.2s ease !important; }
.stTabs [aria-selected="true"] { color:var(--text) !important; border-bottom-color:var(--violet2) !important; background:transparent !important; }
.stTabs [data-baseweb="tab"]:hover { color:var(--text2) !important; background:rgba(255,255,255,0.02) !important; }
.stTabs [data-baseweb="tab-panel"] { padding:2rem 0 0 !important; }

.stTextArea textarea { background:var(--surf2) !important; border:1.5px solid var(--border2) !important; border-radius:14px !important; color:var(--text) !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:1rem !important; padding:1rem 1.1rem !important; line-height:1.65 !important; transition:all 0.25s ease !important; caret-color:var(--violet2) !important; resize:none !important; }
.stTextArea textarea::placeholder { color:var(--muted) !important; font-style:italic !important; }
.stTextArea textarea:focus { border-color:var(--violet) !important; box-shadow:0 0 0 4px rgba(139,111,199,0.14), 0 8px 32px rgba(0,0,0,0.4) !important; background:var(--surf3) !important; }
.stTextArea label { display:none !important; }

[data-testid="stFileUploader"] { background:var(--surf2) !important; border:1.5px dashed var(--border2) !important; border-radius:12px !important; padding:0.9rem 1rem !important; }
[data-testid="stFileUploader"] label { color:var(--muted) !important; font-size:0.85rem !important; }

/* Buttons base */
.stButton > button { font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:700 !important; font-size:0.9rem !important; border-radius:10px !important; height:2.6rem !important; transition:all 0.2s ease !important; cursor:pointer !important; }

/* Classify — gradient primary */
div[data-testid="column"]:first-child .stButton > button { background:linear-gradient(135deg,var(--violet) 0%,var(--pink) 100%) !important; color:#fff !important; border:none !important; padding:0 1.6rem !important; box-shadow:0 4px 20px rgba(139,111,199,0.4) !important; }
div[data-testid="column"]:first-child .stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(224,102,168,0.5) !important; filter:brightness(1.08) !important; }

/* Clear — ghost */
div[data-testid="column"]:nth-child(2) .stButton > button { background:transparent !important; color:var(--muted) !important; border:1.5px solid var(--border2) !important; padding:0 1.2rem !important; }
div[data-testid="column"]:nth-child(2) .stButton > button:hover { color:var(--text2) !important; border-color:var(--muted) !important; background:rgba(255,255,255,0.03) !important; transform:translateY(-1px) !important; }

/* Download */
.stDownloadButton > button { background:transparent !important; color:var(--cyan) !important; border:1.5px solid rgba(56,217,192,0.35) !important; border-radius:10px !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.87rem !important; font-weight:600 !important; transition:all 0.2s ease !important; }
.stDownloadButton > button:hover { background:rgba(56,217,192,0.08) !important; border-color:var(--cyan) !important; transform:translateY(-1px) !important; box-shadow:0 4px 16px rgba(56,217,192,0.2) !important; }

/* Quick Example buttons — neutral, no color */
div[data-testid="stButton-ex_0"] > button,
div[data-testid="stButton-ex_1"] > button,
div[data-testid="stButton-ex_2"] > button,
div[data-testid="stButton-ex_3"] > button,
div[data-testid="stButton-ex_4"] > button,
div[data-testid="stButton-ex_5"] > button,
div[data-testid="stButton-ex_6"] > button {
  height: 2.1rem !important;
  min-height: 2.1rem !important;
  padding: 0 0.85rem !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  letter-spacing: 0.01em !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  transform: none !important;
  filter: none !important;
  line-height: 2.1rem !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text2) !important;
}
div[data-testid="stButton-ex_0"] > button:hover,
div[data-testid="stButton-ex_1"] > button:hover,
div[data-testid="stButton-ex_2"] > button:hover,
div[data-testid="stButton-ex_3"] > button:hover,
div[data-testid="stButton-ex_4"] > button:hover,
div[data-testid="stButton-ex_5"] > button:hover,
div[data-testid="stButton-ex_6"] > button:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: var(--muted) !important;
  color: var(--text) !important;
  box-shadow: none !important;
  transform: none !important;
  filter: none !important;
}


/* HERO */
.hero { padding:3.5rem 0 2.5rem; text-align:center; }
.hero-pill { display:inline-flex; align-items:center; gap:0.5rem; background:rgba(139,111,199,0.13); border:1px solid rgba(169,142,232,0.28); border-radius:99px; padding:0.35rem 1rem; font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--violet2); margin-bottom:1.4rem; font-family:'JetBrains Mono',monospace; }
.hero-pill-dot { width:6px; height:6px; border-radius:50%; background:var(--cyan); box-shadow:0 0 8px var(--cyan); animation:blink 2s ease infinite; }
@keyframes blink { 0%,100%{opacity:1;box-shadow:0 0 8px var(--cyan);}50%{opacity:0.35;box-shadow:0 0 3px var(--cyan);} }
.hero-h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.4rem,5vw,3.8rem); font-weight:700; line-height:1.1; letter-spacing:-0.03em; margin-bottom:1rem; }
.hero-line1 { display:block; color:var(--text); }
.hero-line2 { display:block; background:linear-gradient(120deg,var(--violet2) 0%,var(--pink) 40%,var(--cyan) 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; background-size:200% auto; animation:shimmer 4s linear infinite; }
@keyframes shimmer { 0%{background-position:0% center;} 100%{background-position:200% center;} }
.hero-sub { font-size:1.05rem; color:var(--muted); max-width:500px; margin:0 auto 2rem; line-height:1.65; }
.hero-stats { display:flex; justify-content:center; gap:2.5rem; flex-wrap:wrap; }
.hstat-val { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--violet2),var(--pink)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1; }
.hstat-lbl { font-size:0.7rem; color:var(--muted); margin-top:0.25rem; letter-spacing:0.07em; text-transform:uppercase; font-weight:500; }

.sec-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:500; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.75rem; display:flex; align-items:center; gap:0.6rem; }
.sec-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border2),transparent); }

.result-card { border-radius:18px; padding:1.8rem 2rem; margin:1.6rem 0 1.2rem; border:1.5px solid var(--border2); background:var(--surf2); position:relative; overflow:hidden; animation:cardIn 0.4s cubic-bezier(0.16,1,0.3,1); }
@keyframes cardIn { from{opacity:0;transform:translateY(18px) scale(0.97);} to{opacity:1;transform:translateY(0) scale(1);} }
.result-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.result-card::after { content:''; position:absolute; inset:0; pointer-events:none; background:radial-gradient(ellipse 60% 40% at 80% 110%, var(--glow,rgba(139,111,199,0.12)), transparent 70%); }
.result-header { display:flex; align-items:center; gap:1.2rem; position:relative; z-index:1; }
.result-icon { width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:1.6rem; flex-shrink:0; border:1px solid rgba(255,255,255,0.08); }
.result-level { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; letter-spacing:-0.03em; line-height:1; }
.result-sublabel { font-size:0.73rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; opacity:0.6; margin-top:0.25rem; font-family:'JetBrains Mono',monospace; }
.result-desc { position:relative; z-index:1; margin-top:1.2rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.06); font-size:0.93rem; line-height:1.7; color:var(--text2); }
.conf-row { position:relative; z-index:1; display:flex; align-items:center; gap:1rem; margin-top:1.2rem; }
.conf-num { font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; line-height:1; flex-shrink:0; }
.conf-track { flex:1; background:rgba(255,255,255,0.07); border-radius:99px; height:6px; overflow:hidden; }
.conf-fill { height:100%; border-radius:99px; }
.conf-tag { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; flex-shrink:0; }

.level-k1{border-color:rgba(91,141,238,0.4);--glow:rgba(91,141,238,0.12);}
.level-k1::before{background:linear-gradient(90deg,#3b5bdb,#6a8cff);}
.level-k1 .result-icon{background:rgba(91,141,238,0.12);}
.level-k2{border-color:rgba(62,207,142,0.4);--glow:rgba(62,207,142,0.12);}
.level-k2::before{background:linear-gradient(90deg,#16a34a,#3ecf8e);}
.level-k2 .result-icon{background:rgba(62,207,142,0.12);}
.level-k3{border-color:rgba(247,127,78,0.4);--glow:rgba(247,127,78,0.12);}
.level-k3::before{background:linear-gradient(90deg,#ea580c,#f77f4e);}
.level-k3 .result-icon{background:rgba(247,127,78,0.12);}
.level-k4{border-color:rgba(232,85,154,0.4);--glow:rgba(232,85,154,0.12);}
.level-k4::before{background:linear-gradient(90deg,#db2777,#e8559a);}
.level-k4 .result-icon{background:rgba(232,85,154,0.12);}
.level-k5{border-color:rgba(159,122,234,0.4);--glow:rgba(159,122,234,0.12);}
.level-k5::before{background:linear-gradient(90deg,#7c3aed,#9f7aea);}
.level-k5 .result-icon{background:rgba(159,122,234,0.12);}
.level-k6{border-color:rgba(245,200,66,0.4);--glow:rgba(245,200,66,0.12);}
.level-k6::before{background:linear-gradient(90deg,#d97706,#f5c842);}
.level-k6 .result-icon{background:rgba(245,200,66,0.12);}
.level-nb{border-color:rgba(107,114,128,0.3);--glow:rgba(107,114,128,0.08);}
.level-nb::before{background:linear-gradient(90deg,#374151,#6b7280);}
.level-nb .result-icon{background:rgba(107,114,128,0.1);}

.prob-row{display:flex;align-items:center;gap:0.8rem;margin-bottom:0.7rem;}
.prob-lbl{font-size:0.79rem;width:115px;flex-shrink:0;color:var(--muted);font-weight:500;}
.prob-lbl.top{color:var(--text);font-weight:700;}
.prob-track{flex:1;background:rgba(255,255,255,0.06);border-radius:99px;height:5px;overflow:hidden;}
.prob-fill{height:100%;border-radius:99px;}
.prob-pct{font-family:'JetBrains Mono',monospace;font-size:0.73rem;width:42px;text-align:right;color:var(--muted);}
.prob-pct.top{color:var(--gold);font-weight:600;}

.pyr-row{display:flex;align-items:center;gap:0.85rem;padding:0.75rem 1rem;border-radius:12px;border:1px solid transparent;background:var(--surf2);margin-bottom:0.4rem;transition:all 0.2s ease;cursor:default;}
.pyr-row:hover{background:var(--surf3);transform:translateX(4px);}
.pyr-icon{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.pyr-k{font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;flex-shrink:0;width:24px;}
.pyr-name{font-size:0.88rem;font-weight:700;line-height:1.1;}
.pyr-verbs{font-size:0.71rem;color:var(--muted);margin-top:0.15rem;line-height:1.4;}

.sb-brand{display:flex;align-items:center;gap:0.8rem;padding-bottom:1.4rem;border-bottom:1px solid var(--border);margin-bottom:1.4rem;}
.sb-icon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,var(--violet),var(--pink));display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 14px rgba(139,111,199,0.35);}
.sb-name{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:var(--text);letter-spacing:-0.01em;}
.sb-tagline{font-size:0.68rem;color:var(--muted);margin-top:0.1rem;}
.sb-sec{font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin:1.4rem 0 0.65rem;}
.sb-lv{display:flex;align-items:center;gap:0.65rem;padding:0.45rem 0.65rem;border-radius:8px;margin-bottom:0.2rem;transition:background 0.15s;}
.sb-lv:hover{background:rgba(255,255,255,0.03);}
.sb-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.sb-k{font-family:'JetBrains Mono',monospace;font-size:0.74rem;font-weight:600;width:20px;}
.sb-kn{font-size:0.8rem;color:var(--text2);}

.h-row{display:flex;align-items:center;gap:1rem;padding:0.9rem 1.1rem;border-radius:12px;background:var(--surf2);border:1px solid var(--border);margin-bottom:0.5rem;transition:all 0.2s ease;}
.h-row:hover{border-color:var(--border2);background:var(--surf3);}
.h-q{flex:1;font-size:0.87rem;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;}
.h-badge{display:inline-flex;align-items:center;gap:0.35rem;padding:0.28rem 0.75rem;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:600;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);white-space:nowrap;flex-shrink:0;}
.h-conf{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--muted);flex-shrink:0;min-width:40px;text-align:right;}

.m-card{background:var(--surf2);border:1px solid var(--border);border-radius:14px;padding:1.2rem;text-align:center;transition:all 0.2s ease;}
.m-card:hover{transform:translateY(-2px);}
.m-val{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;line-height:1;}
.m-lbl{font-size:0.73rem;color:var(--muted);margin-top:0.3rem;font-weight:500;}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;color:var(--muted);text-align:center;}
.empty-icon{font-size:3rem;margin-bottom:1rem;opacity:0.35;}
.empty-text{font-size:0.95rem;font-weight:600;}
.empty-hint{font-size:0.78rem;margin-top:0.4rem;color:var(--border2);}

.stAlert{background:rgba(139,111,199,0.08) !important;border:1px solid rgba(169,142,232,0.2) !important;border-radius:10px !important;color:var(--text2) !important;}
</style>
""", unsafe_allow_html=True)

BLOOM_KEYWORDS = {
    'K1': ['define', 'list', 'name', 'state', 'identify', 'label', 'recall', 'recite',
           'what is', 'who is', 'when did', 'where is', 'how many', 'how much',
           'draw and label', 'make a list', 'how many bits', 'how many bytes',
           'how many planets', 'how many bones', 'how many chambers'],
    'K2': ['explain', 'describe', 'summarize', 'paraphrase', 'interpret', 'discuss',
           'in your own words', 'what happens when', 'what is meant by', 'the function of',
           'classify the following'],
    'K3': ['calculate', 'solve', 'apply', 'demonstrate', 'use', 'implement', 'construct',
           'find the', 'compute', 'show how', 'illustrate how', 'employ', 'manipulate',
           'write a function', 'write a program', 'write a code', 'write code',
           'write a python', 'write a script', 'write an algorithm'],
    'K4': ['analyze', 'compare', 'contrast', 'examine', 'differentiate', 'distinguish',
           'compare and contrast', 'similarities and differences', 'break down',
           'relationship between', 'how does', 'what factors'],
    'K5': ['evaluate', 'assess', 'judge', 'critique', 'justify', 'defend', 'argue',
           'which is better', 'which is more effective', 'what is the best',
           'strengths and weaknesses', 'recommend', 'is it ethical'],
    'K6': ['create', 'design', 'develop', 'compose', 'formulate', 'devise', 'invent',
           'propose', 'construct a new', 'develop a plan', 'write a story',
           'design an experiment', 'build a model', 'create a program'],
}

def extract_keyword_features(texts):
    features = []
    for text in texts:
        text_lower = str(text).lower()
        feat_dict = {}
        for level, keywords in BLOOM_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            feat_dict[f'{level}_keywords'] = count
            feat_dict[f'{level}_has_keyword'] = 1 if count > 0 else 0
        feat_dict['has_question_mark'] = 1 if '?' in text else 0
        feat_dict['starts_with_wh'] = 1 if any(text_lower.startswith(w) for w in ['what', 'where', 'when', 'who', 'why', 'how']) else 0
        feat_dict['word_count'] = len(text_lower.split())
        not_blooms_patterns = [
            'how are you', 'how are you doing', 'what time is it', 'good morning',
            'good afternoon', 'good evening', 'good night', 'hello', 'hi there',
            'bring me', 'pass me', 'give me', 'close the', 'open the',
            'turn on', 'turn off', 'wait here', 'sit down', 'stand up',
            'i am feeling', 'i feel', 'that was', 'that is',
            'nearest bus', 'nearest atm', 'nearest bank', 'nearest hospital',
            'where can i find', 'how do i get', 'which way to', 'where should i',
        ]
        feat_dict['has_not_blooms_pattern'] = 1 if any(p in text_lower for p in not_blooms_patterns) else 0
        command_words = ['please', 'bring', 'close', 'open', 'turn', 'sit', 'stand', 'wait', 'call', 'send']
        feat_dict['is_command'] = 1 if any(text_lower.startswith(w) or text_lower.startswith('please ' + w) for w in command_words) else 0
        features.append(list(feat_dict.values()))
    return np.array(features)

LEVEL_INFO = {
    "K1": {"name":"Remember","color":"#5b8dee","css_class":"level-k1","icon":"📌","desc":"Recall facts and basic concepts. The student retrieves relevant knowledge from long-term memory.","verbs":"Define, List, Name, State, Identify, Label, Recall"},
    "K2": {"name":"Understand","color":"#3ecf8e","css_class":"level-k2","icon":"💡","desc":"Explain ideas or concepts. The student constructs meaning from instructional messages.","verbs":"Explain, Describe, Summarize, Interpret, Discuss, Classify"},
    "K3": {"name":"Apply","color":"#f77f4e","css_class":"level-k3","icon":"🔧","desc":"Use information in new situations. The student carries out a procedure in a given situation.","verbs":"Calculate, Solve, Apply, Demonstrate, Use, Implement"},
    "K4": {"name":"Analyze","color":"#e8559a","css_class":"level-k4","icon":"🔍","desc":"Draw connections among ideas. The student breaks material into parts and determines their relationships.","verbs":"Analyze, Compare, Contrast, Examine, Differentiate, Break down"},
    "K5": {"name":"Evaluate","color":"#9f7aea","css_class":"level-k5","icon":"⚖️","desc":"Justify a decision or course of action. The student makes judgments based on criteria and standards.","verbs":"Evaluate, Assess, Judge, Critique, Justify, Defend"},
    "K6": {"name":"Create","color":"#f5c842","css_class":"level-k6","icon":"✨","desc":"Produce new or original work. The student puts elements together to form a new, coherent whole.","verbs":"Create, Design, Develop, Compose, Formulate, Devise"},
    "not_blooms": {"name":"Not Blooms","color":"#9ca3af","css_class":"level-nb","icon":"🚫","desc":"This does not appear to be a Bloom's Taxonomy educational question. It may be a greeting, command, or non-academic query.","verbs":"N/A"},
}

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "blooms_model.pkl")
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(model, question):
    processed = preprocess(question)
    prediction = model.predict([processed])[0]
    try:
        proba = model.predict_proba([processed])[0]
        classes = model.classes_
        confidence = dict(zip(classes, proba))
        top_confidence = confidence[prediction]
    except:
        confidence = {}
        top_confidence = 1.0
    return prediction, top_confidence, confidence

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">🎓</div>
        <div>
            <div class="sb-name">Bloom's Classifier</div>
            <div class="sb-tagline">Anderson & Krathwohl, 2001</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Cognitive Levels</div>', unsafe_allow_html=True)
    for lvl in ["K6","K5","K4","K3","K2","K1","not_blooms"]:
        info = LEVEL_INFO[lvl]
        display_key = "NB" if lvl == "not_blooms" else lvl
        st.markdown(f"""
        <div class="sb-lv">
            <div class="sb-dot" style="background:{info['color']};box-shadow:0 0 5px {info['color']}88;"></div>
            <div class="sb-k" style="color:{info['color']};">{display_key}</div>
            <div class="sb-kn">{info['name']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("↺  Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

st.markdown("""
<div class="hero">
    <div class="hero-pill">
        <div class="hero-pill-dot"></div>
        AI-Powered · 93.18% Accuracy
    </div>
    <h1 class="hero-h1">
        <span class="hero-line1">Classify Any Question</span>
        <span class="hero-line2">Across Bloom's Taxonomy</span>
    </h1>
    <p class="hero-sub">Instantly identify the cognitive level of educational questions — from simple recall all the way to creative synthesis.</p>
    <div class="hero-stats">
        <div><div class="hstat-val">7</div><div class="hstat-lbl">Cognitive Levels</div></div>
        <div><div class="hstat-val">93%</div><div class="hstat-lbl">Accuracy</div></div>
        <div><div class="hstat-val">&lt;1s</div><div class="hstat-lbl">Response Time</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("⚠️ Model not found at `models/blooms_model.pkl`. Please ensure the model file exists.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["✦  Classify", "⊞  Batch", "◷  History"])

with tab1:
    col_main, col_ref = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown('<div class="sec-label">Your Question</div>', unsafe_allow_html=True)

        question_input = st.text_area(
            label="q",
            placeholder='Type or paste your educational question here…\ne.g. "Compare the causes and effects of the French Revolution."',
            height=120,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2, _ = st.columns([1.1, 0.9, 3])
        with col_btn1:
            classify_btn = st.button("✦  Classify", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        st.markdown('<div style="margin-top:1.1rem;"><div class="sec-label" style="font-size:0.6rem;">Quick examples</div></div>', unsafe_allow_html=True)

        examples = [
            ("ex_0", "Remember",   "What is the definition of photosynthesis?"),
            ("ex_1", "Understand",  "Explain how the water cycle works."),
            ("ex_2", "Apply",       "Calculate the area of a circle with radius 5 cm."),
            ("ex_3", "Analyze",     "Compare and contrast capitalism and socialism."),
            ("ex_4", "Evaluate",    "Which energy source is more sustainable and why?"),
            ("ex_5", "Create",      "Design an experiment to test water quality."),
            ("ex_6", "Not Blooms",  "Good morning! How are you doing today?"),
        ]

        ex_r1 = st.columns(4)
        ex_r2 = st.columns(4)
        for i, (key, label, q) in enumerate(examples):
            col_group = ex_r1 if i < 4 else ex_r2
            col_idx   = i if i < 4 else i - 4
            with col_group[col_idx]:
                if st.button(label, key=key, use_container_width=True):
                    question_input = q
                    classify_btn = True

        if classify_btn and question_input.strip():
            label, conf, all_conf = predict(model, question_input.strip())
            info = LEVEL_INFO.get(label, LEVEL_INFO["not_blooms"])
            st.session_state.history.append({"question": question_input.strip(), "label": label, "confidence": conf})

            st.markdown(f"""
            <div class="result-card {info['css_class']}">
                <div class="result-header">
                    <div class="result-icon">{info['icon']}</div>
                    <div>
                        <div class="result-level" style="color:{info['color']};">{label} — {info['name']}</div>
                        <div class="result-sublabel" style="color:{info['color']};">Bloom's Cognitive Level</div>
                    </div>
                </div>
                <div class="result-desc">{info['desc']}</div>
                <div class="conf-row">
                    <div class="conf-num" style="color:{info['color']};">{conf*100:.0f}%</div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{conf*100:.0f}%;background:linear-gradient(90deg,{info['color']},{info['color']}bb);"></div>
                    </div>
                    <div class="conf-tag">Confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if all_conf:
                st.markdown('<div style="margin-top:1.5rem;"><div class="sec-label">Probability Breakdown</div></div>', unsafe_allow_html=True)
                for lv, prob in sorted(all_conf.items(), key=lambda x: x[1], reverse=True):
                    lv_info = LEVEL_INFO.get(lv, LEVEL_INFO["not_blooms"])
                    is_top = lv == label
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-lbl {'top' if is_top else ''}">{lv} — {lv_info['name']}</div>
                        <div class="prob-track"><div class="prob-fill" style="width:{prob*100:.0f}%;background:{lv_info['color']};opacity:{'1' if is_top else '0.28'};"></div></div>
                        <div class="prob-pct {'top' if is_top else ''}">{prob*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif classify_btn and not question_input.strip():
            st.warning("Please enter a question to classify.")

    with col_ref:
        st.markdown('<div class="sec-label">Bloom\'s Pyramid</div>', unsafe_allow_html=True)
        for lvl in ["K6","K5","K4","K3","K2","K1","not_blooms"]:
            info = LEVEL_INFO[lvl]
            display_key = "NB" if lvl == "not_blooms" else lvl
            st.markdown(f"""
            <div class="pyr-row" style="border-color:{info['color']}22;">
                <div class="pyr-icon" style="background:{info['color']}18;">{info['icon']}</div>
                <div class="pyr-k" style="color:{info['color']};">{display_key}</div>
                <div>
                    <div class="pyr-name" style="color:{info['color']};">{info['name']}</div>
                    <div class="pyr-verbs">{info['verbs']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="sec-label">Batch Classification</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.84rem;color:#6e6a90;margin-bottom:1.2rem;">Enter one question per line, or upload a CSV with a <code style="background:rgba(255,255,255,0.07);padding:0.1rem 0.4rem;border-radius:4px;font-size:0.78rem;font-family:\'JetBrains Mono\',monospace;">Questions</code> column.</p>', unsafe_allow_html=True)

    bcol1, _ = st.columns([3, 2], gap="large")
    with bcol1:
        batch_text = st.text_area("b", placeholder="What is the capital of France?\nExplain how photosynthesis works.\nDesign an experiment to measure reaction speed.", height=200, label_visibility="collapsed")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        run_batch = st.button("⊞  Run Batch Classification →")

    if run_batch:
        questions_list = []
        if uploaded_csv:
            df_upload = pd.read_csv(uploaded_csv)
            if "Questions" in df_upload.columns:
                questions_list = df_upload["Questions"].dropna().tolist()
            else:
                st.error("CSV must have a 'Questions' column.")
        elif batch_text.strip():
            questions_list = [q.strip() for q in batch_text.strip().split("\n") if q.strip()]

        if questions_list:
            results = []
            for q in questions_list:
                lbl, conf, _ = predict(model, q)
                results.append({"Question": q, "Level": lbl, "Name": LEVEL_INFO.get(lbl,{}).get("name",""), "Confidence": f"{conf*100:.1f}%"})
                st.session_state.history.append({"question": q, "label": lbl, "confidence": conf})

            df_results = pd.DataFrame(results)
            st.markdown(f'<div style="margin-top:1.6rem;"><div class="sec-label">Results — {len(results)} questions</div></div>', unsafe_allow_html=True)

            from collections import Counter
            dist = Counter(r["Level"] for r in results)
            mcols = st.columns(min(len(dist), 4))
            for i, (lv, cnt) in enumerate(sorted(dist.items())):
                info = LEVEL_INFO.get(lv, LEVEL_INFO["not_blooms"])
                with mcols[i % 4]:
                    st.markdown(f'<div class="m-card" style="border-color:{info["color"]}28;"><div class="m-val" style="color:{info["color"]};">{cnt}</div><div class="m-lbl">{lv} · {info["name"]}</div></div>', unsafe_allow_html=True)

            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
            for r in results:
                info = LEVEL_INFO.get(r["Level"], LEVEL_INFO["not_blooms"])
                st.markdown(f'<div class="h-row"><div class="h-q">{r["Question"][:100]}{"…" if len(r["Question"])>100 else ""}</div><div class="h-badge" style="color:{info["color"]};border-color:{info["color"]}44;">{info["icon"]} {r["Level"]}</div><div class="h-conf">{r["Confidence"]}</div></div>', unsafe_allow_html=True)

            st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
            st.download_button("⬇  Download Results CSV", data=df_results.to_csv(index=False), file_name="blooms_results.csv", mime="text/csv")

with tab3:
    st.markdown('<div class="sec-label">Classification History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<div class="empty"><div class="empty-icon">◷</div><div class="empty-text">No history yet</div><div class="empty-hint">Classified questions will appear here</div></div>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history):
            info = LEVEL_INFO.get(item["label"], LEVEL_INFO["not_blooms"])
            st.markdown(f'<div class="h-row"><div class="h-q">{item["question"]}</div><div class="h-badge" style="color:{info["color"]};border-color:{info["color"]}44;">{info["icon"]} {item["label"]} — {info["name"]}</div><div class="h-conf">{item["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["confidence"] = hist_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        hist_df.columns = ["Question", "Level", "Confidence"]
        st.download_button("⬇  Export History CSV", data=hist_df.to_csv(index=False), file_name="blooms_history.csv", mime="text/csv")