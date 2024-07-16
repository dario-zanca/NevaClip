import clip
from NeVA import NeVAWrapper
import utils
from PIL import Image
from capmit1003 import CapMIT1003

def main():

    # Configurations
    clip_backbone = 'RN50'
    model_path = "Models/"
    device = "cpu" # or "cuda"
    image_size = 224
    foveation_sigma = 0.2
    blur_filter_size = 41
    blur_sigma = 10
    lr = 0.1

    # Load CLIP
    print("Downloading CLIP...")
    model, preprocess = clip.load(
        clip_backbone,
        device=device,
        download_root=model_path
    )
    vision_model = model.visual
    text_model = model.encode_text

    # Create NevaClip Model
    print("Creating NevaClip wrapper...")
    NeVA_model = NeVAWrapper(
        downstream_model=vision_model,
        criterion=utils.cosine_sim,
        target_function=utils.target_function,
        image_size=image_size,
        foveation_sigma=foveation_sigma,
        blur_filter_size=blur_filter_size,
        blur_sigma=blur_sigma,
        forgetting=0,
        foveation_aggregation=1,
        device=device
    )

    with CapMIT1003('capmit1003.db') as db:

        CapMIT1003.download_images()

        image_captions = db.get_captions()

        for pair in image_captions.itertuples(index=False):

            original_image = Image.open(pair.img_path)
            preprocessed_image = preprocess(original_image).unsqueeze(0)
            if device == "cuda":
                preprocessed_image = preprocessed_image.cuda()
            caption = pair.caption

            # Generate a scanpath with NevaClip
            current_scanpath = NeVA_model.run_optimization(
                preprocessed_image,
                text_model(clip.tokenize([caption]).to(device)),
                scanpath_length=10,
                opt_iterations=20,
                learning_rate=lr
            )

            print(current_scanpath)  # relative coordinates

if __name__ == '__main__':
    main()
