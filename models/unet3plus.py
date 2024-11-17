import torch
import torch.nn as nn
import torch.nn.functional as F

class ULayerDown(nn.Module):
    def __init__(self,channels_in,channels_out, **layer_kwargs):
        super().__init__()
        self.conv_layer = nn.Conv2d(channels_in,channels_out,3,**layer_kwargs)
        self.activation = nn.ReLU()
        self.scale_layer = nn.MaxPool2d(2,padding=1)
        self.output = None

    def forward(self,X) :
            Y = self.conv_layer(X)
            Y = self.activation(Y)
            Y = nn.Dropout(0.4)(Y)
            Y = self.scale_layer(Y)
            self.output = Y

            return Y


class ULayerUp(nn.Module) :
    def __init__(self,single_feature_map_channels,n_classes,mask_shape,connected_nodes,_device="cuda", **layer_kwargs):
        super().__init__()
        self.channels_per_fmap = single_feature_map_channels
        self.conv_layer = nn.Conv2d(self.channels_per_fmap*(len(connected_nodes)+1), self.channels_per_fmap, 3,padding=1, **layer_kwargs)
        self.normed = nn.BatchNorm2d(self.channels_per_fmap)
        self.activation = nn.ReLU()
        self.scale_layer = nn.Upsample(scale_factor=2)
        self.output = None
        self.device = _device
        self.prep_layers = []
        self.sources = connected_nodes
        self.side_mask_predictor = nn.Sequential(
            nn.Conv2d(self.channels_per_fmap,n_classes,3),
            nn.Sigmoid(),
            nn.Upsample(mask_shape)
        )
        self.side_mask_output = None
        for node in connected_nodes :
            self.prep_layers.append(nn.Conv2d(node.conv_layer.out_channels, self.channels_per_fmap,3,padding=1).to(self.device))

    def forward(self,X) :
        to_cat = [X]
        for i in range(len(self.sources )) :
            resized_output = nn.Upsample(X.shape[-2:],mode="nearest")(self.sources[i].output)
            Xc = self.prep_layers[i](resized_output)
            to_cat.append(Xc)
        in_tensor = torch.cat(to_cat,1)
        Y = self.conv_layer(in_tensor)
        Y = self.normed(Y)
        Y = self.activation(Y)
        Y = nn.Dropout(0.4)(Y)
        self.output = Y
        if  self.training and self.scale_layer is not None and self.side_mask_predictor is not None:
            Y = self.scale_layer(Y)
            self.side_mask_output = self.side_mask_predictor(Y)

        return Y


class ClassificationArm(nn.Module) :
    def __init__(self,in_channels,n_classes,input_size=25):
        super().__init__()
        pooled_size = input_size//2
        self.sequence = nn.Sequential(
            nn.Upsample([input_size,input_size]),
            nn.Conv2d(in_channels,n_classes,3,padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(pooled_size),
            nn.Flatten(start_dim=1),
            nn.Linear(n_classes*pooled_size*pooled_size,96),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(96,64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,n_classes)
        )
        self.sig_layer = nn.Sigmoid()

    def forward(self,X):
        Y = self.sequence(X)

        return self.sig_layer(Y)


class UNet3plus(nn.Module) :
    def __init__(self, n_classes = 2, in_channels = 3, depth = 3, first_output_channels = 8, upwards_feature_channels = 16, sideways_mask_shape = [200,200]):
        super().__init__()
        input_layer = ULayerDown(in_channels,first_output_channels)
        down_layers = [ULayerDown(first_output_channels*2**i,first_output_channels*2**(i+1)) for i in range(depth-1)]
        down_layers.append(ULayerDown(first_output_channels*2**(depth-1),upwards_feature_channels))
        down_layers.insert(0,input_layer)
        self.seqdown = nn.Sequential(*down_layers)
        up_layers = []
        for i in range(depth) :
            up_layers.append(
                ULayerUp(upwards_feature_channels, n_classes, mask_shape = sideways_mask_shape, connected_nodes= down_layers[:depth-1-i]+ up_layers )
            )
        self.sequp = nn.Sequential(*up_layers)
        self.classifier = ClassificationArm(upwards_feature_channels,n_classes)
        self.last_layer = nn.Conv2d(upwards_feature_channels,n_classes,3,padding=1)
        self._side_mask_shape = sideways_mask_shape
        self.presence_prediction = None

    def forward(self,X):
        Y = nn.functional.normalize(X,dim=1)
        Y = self.seqdown(Y)
        Y = nn.Upsample(scale_factor=2)(Y)
        self.presence_prediction = self.classifier(Y)
        Y = self.sequp(Y)
        for layer in self.sequp :
            layer.side_mask_output *= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)
        Y = self.last_layer(Y)
        Y*= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)
        Y = nn.Upsample(X.shape[-2:])(Y)

        return nn.Softmax(dim=1)(Y)
