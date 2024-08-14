from llava.train.train import train
import PIL

# hot fix for image size exceeds limit
PIL.Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
