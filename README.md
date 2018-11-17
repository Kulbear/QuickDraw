## Doodle Classification Note

Training config:
- ~ 75% of 30k / class ( like ~23k / class)
- size 128 x 128
- batch size 128 (can be a bit more larger)
- 500, steps every epoch, so every epoch train with 128 * 500 images
- Adam, lr start at 2e-4, use cyclic LR with cos annealing 
```python
def cos_annealing_lr(initial_lr, cur_epoch, epoch_per_cycle):
    return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2
```
- 6 cycles, each cycle 25 epochs. By checking the training log I think this schedule should be improved further)

Model used (with best score):
- MobileNet 0.913 (by hflip TTA)
- ResNet18 0.919
- Xception 0.925 (not with the current 3 channel information, I think the current one can give a better result)
- SE-ResNeXt50 (32 x 4d) 0.930

Ideas used: 
- imagenet-pretrained weights
- larger images
    I tried Xception with 224 x 224 image, by using about 20k/class, and just ReduceOnPlaque can get 0.925, I think larger --> better
- grayscale strokes to ordered strokes
- country code information encoding 
```python
c_value = country_mapping.get(df.iloc[i]['countrycode'], 40)
x[i, :, :, 0] = img + mask * c_value
```
- highlight first and last stokes (may drop the last and retry)
```python
if i == 0 or i == len(stroke[0]) - 2:
    color = 255
    _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                 (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
```
- hflip TTA (good when LB < 0.920, bad when LB >= 0.920)
- last few layers in the model, with dropout rate 0.075 (prob not good)
```python
    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
```

Possible Models:

Size 1:

- Xception
- ResNet50
- SE ResNeXt50

Size 2:
- SE ResNeXt101
- DPN?
- InceptionResV2


TODO:
- better LR scheduler (this seems to be the most important one)
- update network after several batches to increase the valid batch size
- finetune with different optimizers?
- larger batches
- lstm
    - bi-lstm
    - gru?
    - cnn as feature extractor and rnn accept result from global pooling?
- label denoising