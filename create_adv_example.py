from functools import reduce
import torch as ch
import numpy as np
import os
import cv2
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import torchvision.models as models

from cameraWarping import get_warped_images_torch


mean = ch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
std = ch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
mean, std = mean.cuda(), std.cuda()


def unprocess(tensor):
    for i in range(len(tensor)):
        tensor[i] = (tensor[i] * std) + mean
    return tensor


def preprocess(tensor):
    for i in range(len(tensor)):
        tensor[i] = (tensor[i] - mean) / std
    return tensor


def load_images(folder):
    images = []
    for fp in os.listdir(folder):
        src = cv2.imread(os.path.join(folder, fp))
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        images.append(src)
    images = np.array(images).transpose(0, 3, 1, 2)
    images = ch.from_numpy(images)
    return images


def get_patch(model, base_image, patch_corners, patch_size, iters=1000):
    # Get adversarial patch ready
    adv_patch = ch.rand((3, patch_size, patch_size)).cuda()
    adv_patch = Variable(adv_patch.clone(), requires_grad=True)
    adv_patch = adv_patch.cuda()

    # Define optimizer and losses
    optimizer = ch.optim.Adam([adv_patch], lr=0.1)
    iterator = range(iters)
    loss_fn = ch.nn.CrossEntropyLoss().cuda()
    loss_fn_all = ch.nn.CrossEntropyLoss(reduction='none').cuda()

    # Get torch version of OG image
    base_image_pt = ch.from_numpy(base_image)

    # Get original predictions
    _, __, og_batch = get_warped_images_torch(base_image_pt)
    og_batch = og_batch.cuda()

    # from cameraWarping import ch_to_cv2
    # os.mkdir("./test")
    # for i in range(len(og_batch)):
    #     ii = ch_to_cv2(og_batch[i].numpy().astype('uint8'))
    #     cv2.imwrite("./test/%d.png" % (i), ii)

    y_og = model(preprocess(og_batch)).argmax(1)
    iterator = tqdm(range(iters))

    for i in iterator:
        optimizer.zero_grad()

        # Sew patch on top of original image
        img_mod = base_image_pt.clone()

        img_mod[:, patch_corners[0]:patch_corners[0]+patch_size,
                patch_corners[1]:patch_corners[1]+patch_size, ] = adv_patch

        # Get warped differentiable images
        distances, degrees, batch_mod = get_warped_images_torch(img_mod)
        batch_mod = batch_mod.cuda()
        new_output = model(preprocess(batch_mod))

        # Maximize loss w.r.t original predictions
        loss = -loss_fn(new_output, y_og)
        print_loss = -loss.item()

        iterator.set_description('Loss : %f' % print_loss)
        # loss.backward(retain_graph=True)
        loss.backward()

        optimizer.step()
        adv_patch.data = ch.clip(adv_patch.data, 0, 1)
    
    all_losses = loss_fn_all(new_output, y_og).detach().cpu()
    return adv_patch.detach(), img_mod.detach(), all_losses, distances, degrees


if __name__ == "__main__":
    import sys
    log_num = sys.argv[1]

    # Load base image
    pil_image = Image.open("./strip.jpg").convert('RGB')
    img_original = np.array(pil_image).transpose(2, 0, 1) / 1.

    # Load model
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    model = ch.nn.DataParallel(model)
    model = model.cuda()

    # Get optimal patch, modified image
    patch, modded, loss, distances, degrees = get_patch(model, img_original, (180, 130), 30, iters=200)

    # Dump information to file
    with open("logs/%s.csv" % log_num, 'w') as f:
        for d, l in zip(distances, loss):
            f.write("%d,%.4f\n" % (d, l))

    # Un-normalize image
    # modded = unprocess(modded)

    # modded_cpu = modded.numpy()[0].transpose(1, 2, 0)
    # Clip to 0/1
    # modded_cpu = np.clip(modded_cpu, 0, 1)

    # plt.imshow(modded_cpu)
    # plt.savefig("with_sticker.png")
