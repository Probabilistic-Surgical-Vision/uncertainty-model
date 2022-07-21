# _Randomly Connected Neural Networks for Self-Supervised Monocular Depth Estimation_ Model Template

Original Paper: [https://www.tandfonline.com/doi/full/10.1080/21681163.2021.1997648](https://www.tandfonline.com/doi/full/10.1080/21681163.2021.1997648)

## To-do List

- [x] Copy over all code relating to the best Tukra et al. model
- [x] Reduce args object to independent arguments.
- [ ] Create YAML file of all training/dataset/model/evaluation arguments.
- [x] Remove any unnecessary conditionals and checks in code.
- [x] Combine similar blocks together in code.
- [ ] Comment all classes and methods.

## Notes

Decoder out:
```
input is x1, x2, prev_disp, prev_skip

x1_upsampled <- upsample(x1) { upsample_channels <- x1_channels // divisor }
skip <- se(prev_skip + x1_upsampled) { skip_channels <- prev_skip_channels + x2_channels }
x_concat <- iconv(x1_upsampled + skip + prev_disp? ) { out_channels <- upsample_channels + skip_channels + disp_channels }
disp <- disp_conv(x_concat) { disp_channels <- out_channels }
```