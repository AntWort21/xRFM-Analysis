{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # PROJECT_NOTES.md\
\
Team roles:\
- Yuan Cao: xRFM line\
- Adit: XGBoost / LightGBM / Random Forest\
- Niko: report writing\
\
Project goals:\
- Compare xRFM with strong baselines on 5 tabular datasets\
- Use shared preprocessing and fair train/val/test splits\
- Tune on validation, evaluate once on held-out test\
- Store results in structured files for report writing\
\
Repository design rules:\
- Anything affecting fairness must go in shared modules\
- Model-specific logic should be separated\
- Keep code simple and reproducible\
- Add TODO markers for dataset-specific details}