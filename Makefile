.PHONY: install check exp1b joint reject-ablation context-ablation final-outputs fixed-window all-results

install:
	python -m pip install -r requirements.txt

check:
	python -m compileall diploma_project

exp1b:
	python -m diploma_project.experiments.baseline_trainable_full_token_retrieval_head

joint:
	python -m diploma_project.experiments.joint_compression_reject

reject-ablation:
	python -m diploma_project.experiments.joint_reject_type_ablation

context-ablation:
	python -m diploma_project.experiments.joint_context_reserve_ablation

final-outputs:
	python -m diploma_project.experiments.final_protocol_additions --lightweight

fixed-window:
	python -c "from diploma_project.experiments.final_protocol_additions import make_fixed_window_control; make_fixed_window_control(run_inference=True)"

all-results: exp1b joint reject-ablation context-ablation final-outputs fixed-window

