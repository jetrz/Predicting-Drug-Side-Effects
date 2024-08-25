from code.utils.dataset_processing import generate_se_embedding, generate_soc_data

from code.scripts.mlp_baseline import run_mlp_baseline
from code.scripts.mlp_concat import run_mlp_concat
from code.scripts.mlp_mega_concat import run_mlp_mega_concat
from code.scripts.mlp_concat_w_se_embeds import run_mlp_concat_w_se_embeds
from code.scripts.mlp_concat_w_soc_labels import run_mlp_concat_w_soc_labels
from code.scripts.mlp_mega_concat_cheat import run_mlp_mega_concat_cheat

if __name__ == "__main__":
    run_mlp_baseline()
    # generate_soc_data(type="weighted", randomise="randomised")