#!/usr/bin/env bash
# Regenerate analysis/results/: the printed output of every analysis the paper quotes, one file per table or claim,
# so each number in the paper can be found in a committed file without running anything. Run from the repository
# root; about five minutes on a laptop (Python 3.10+, matplotlib and scipy). Not included: score_2007.py, which
# re-scores all 464 programs on 2007 flights (about six hours on 16 cores; its output is data/2007/scores.csv), and
# analyze.py, which needs the raw run trees.
set -euo pipefail
cd "$(dirname "$0")/.."
R=analysis/results; mkdir -p "$R"
tmp=$(mktemp -d)
run() { local out=$1; shift; echo "\$ $*" > "$R/$out"; "$@" >> "$R/$out" 2>&1; echo "wrote $R/$out"; }
run table3_figure1_study2_pairings.txt  python3 analysis/figures.py data/study2/cells.csv "$tmp/study2"
run counts_ledger.txt                   python3 analysis/ledger.py data/study1/cells.csv data/study2/cells.csv data/study3/cells.csv
run compliance_audit.txt                python3 analysis/audit_exported_code.py
run table5_exclusion_rules.txt          python3 analysis/sensitivity_violations.py data/study2/cells.csv data/study2/final_eval.csv
run table6_best_of_k.txt                python3 analysis/best_of_k.py data/study2/cells.csv data/study2/final_eval.csv
run tables8_13_yield_and_cost.txt       python3 analysis/policy_tables.py data/study2/cells.csv data/study2/final_eval.csv data/study3/cells.csv data/study3/final_eval.csv
run table7_study3_vs_study2.txt         python3 analysis/compare_arms.py data/study3/cells.csv data/study2/cells.csv
run figure2_model_upgrade.txt           python3 analysis/figure_arms.py data/study3/cells.csv data/study2/cells.csv "$tmp/study3"
run table10_interactions.txt            python3 analysis/interaction_table.py data/study2/cells.csv data/study3/cells.csv
run figures3_4_interaction.txt          python3 analysis/figure_interaction.py data/study2/cells.csv data/study3/cells.csv "$tmp/interaction"
run table11_later_year.txt              python3 analysis/analyze_2007.py
run table9_and_text_numbers.txt         python3 analysis/paper_numbers.py data/study2/cells.csv data/study3/cells.csv data/study1/cells.csv
run appendixD_study2_tokens.txt         python3 analysis/token_totals.py
run study1_v4pro_row.txt                python3 analysis/study1_v4pro.py
run section4.9_cache_requests.txt       python3 analysis/cache_requests.py
run table4_figure6_study1.txt           python3 analysis/study1_agents_vs_noise.py data/study1/cells.csv
run table17_best_of_k_by_pairing.txt    python3 analysis/best_of_k_by_pairing.py data/study2/cells.csv data/study2/final_eval.csv
run table18_repricing.txt               python3 analysis/table18_repricing.py
run table12_study1_cache_cost.txt       python3 analysis/table12_study1_cache_cost.py
run table14_decisions.txt               python3 analysis/table14_decisions.py
run rescore_study3_pi_seed35.txt        python3 analysis/rescore.py code/study3/var7/airline/pi/glm-5.3/seed35
rm -rf "$tmp"
