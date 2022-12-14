# print financial performance
evaluate_finperf:
	python src/evaluate_models.py --data-source SPY &&\
	python src/evaluate_models.py --data-source NDAQ

# print financial and time performance
evaluate_models:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source SPY &&\
	python src/evaluate_models.py --time-performance-iterations 100 --data-source NDAQ

# render plots
plot_comps:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source SPY --save-figs &&\
	python src/evaluate_models.py --time-performance-iterations 100 --data-source NDAQ --save-figs

# update perf.csv
update_perf:
	python src/evaluate_models.py --time-performance-iterations 100 --update-perf --data-source SPY &&\
	python src/evaluate_models.py --time-performance-iterations 100 --update-perf --data-source NDAQ

# update all plots and perf.csv
plots_and_perf:
	python src/evaluate_models.py --time-performance-iterations 100  --save-figs --update-perf --data-source SPY &&\
	python src/evaluate_models.py --time-performance-iterations 100  --save-figs --update-perf --data-source NDAQ