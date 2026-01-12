import click

@click.group()
def main():
    pass

@main.command("download_wisard")
def download_wisard():
    import gdown
    output = "dataset/WiSARD.zip"
    file_id = "1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht"
    gdown.download(
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
        output,
        fuzzy=True,
    )


@main.command("preprocess_classification")
@click.option("--output-dir", type=str, required=True, help="Path to the output directory", default="dataset/sard_pose")
def preprocess_classification(output_dir):
    from sarfusion.data.preprocess import generate_pose_classification_dataset
    generate_pose_classification_dataset(output_dir)
 

@main.command("preprocess_wisard")
def preprocess_wisard():
    from sarfusion.data.preprocess import wisard_to_yolo_dataset
    wisard_to_yolo_dataset("dataset/WiSARD")
    
@main.command("annotate_wisard")
@click.option("--model-yaml", type=str, help="Path to the model yaml file", default="parameters/WiSARD_pose/parameters.yaml")
def annotate_wisard(model_yaml):
    from sarfusion.data.preprocess import annotate_rgb_wisard
    root = "dataset/WiSARD"
    annotate_rgb_wisard(root, model_yaml)
    
@main.command("simplify_wisard")
def annotate_wisard():
    from sarfusion.data.preprocess import simplify_wisard
    root = "dataset/WiSARD"
    simplify_wisard(root)
    
    
@main.command("tile_wisard")
@click.option("--data", type=str, help="Path to the data yaml file", default="wisards_vis_all_ir_sync-test_vis.yaml")
def tile_wisard(data):
    from sarfusion.data.preprocess import generate_tiled_wisard
    root = "dataset/WiSARD"
    generate_tiled_wisard(root, data_yaml=data)
    
    
@main.command("experiment")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
@click.option(
    "--parallel", default=False, help="Run the experiments in parallel", is_flag=True
)
@click.option(
    "--yolo", default=False, help="Run the experiments with YOLO workspace", is_flag=True
)
@click.option(
    "--only-create",
    default=False,
    help="Creates params files with running them",
    is_flag=True,
)
def experiment(parameters, parallel, only_create, yolo):
    from sarfusion.experiment.experiment import experiment as run_experiment
    run_experiment(param_path=parameters, parallel=parallel, only_create=only_create, yolo=yolo)


@main.command("run")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def run(parameters):
    from sarfusion.experiment.experiment import run as run_single
    run_single(param_path=parameters)
    
    
@main.command("test")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def test(parameters):
    from sarfusion.experiment.experiment import test as run_test
    run_test(parameters)
    
    
@main.command("yolo")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def yolo(parameters):
    from sarfusion.experiment.run import yolo_train
    yolo_train(parameters)
    
    
@main.command("app")
def app():
    from sarfusion.api.app import run_app
    run_app()
    
    
if __name__ == "__main__":
    main()