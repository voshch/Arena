import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder_mask3(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder_mask3, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(embed_dim + 1, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x, src_key_padding_mask, avg_pool_mask, no_goal_mask):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)
            x = x * avg_mask
        x = torch.mean(x, dim=1)        
        x = x.reshape(x.shape[0], -1)
        if no_goal_mask.sum().item() == 9:
            dev_gpu = no_goal_mask.get_device()
            no_goal_mask = torch.tensor([9]).to(dev_gpu)
        x = torch.cat((x, no_goal_mask.unsqueeze(1)), axis=1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x    

class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            self.num_action_params = 4  # last two dims are the cos and sin of the angle
        else:
            self.num_action_params = 2

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
 

class OmniVLA_edge(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:

        super(OmniVLA_edge, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=9) # context
            self.num_obs_features_map = self.goal_encoder._fc.in_features            
            
            if self.late_fusion:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features_img = self.goal_encoder_img._fc.in_features
            
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
            
        if self.num_obs_features_map != self.obs_encoding_size:
            self.compress_obs_enc_map = nn.Linear(self.num_obs_features_map, self.obs_encoding_size)
        else:
            self.compress_obs_enc_map = nn.Identity()            
                    
        if self.num_goal_features_img != self.goal_encoding_size:
            self.compress_goal_enc_img = nn.Linear(self.num_goal_features_img, self.goal_encoding_size)
        else:
            self.compress_goal_enc_img = nn.Identity()
           
        self.num_goal_features_lan = 4096
        if self.num_goal_features_lan != self.goal_encoding_size:
            self.compress_goal_enc_lan = nn.Linear(self.num_goal_features_lan, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc_lan = nn.Identity()
        
        self.decoder = MultiLayerDecoder_mask3(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2+1+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),            
        )
        
        self.film_model = build_film_model(8, 10, 128, 512)
               
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )           
               
        self.goal_mask_0 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_0[:, -4] = True 
        self.goal_mask_0[:, -2] = True 
        self.goal_mask_0[:, -1] = True                    
        self.goal_mask_1 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_1[:, -3] = True    
        self.goal_mask_1[:, -2] = True   
        self.goal_mask_1[:, -1] = True                   
        self.goal_mask_2 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_2[:, -2] = True   
        self.goal_mask_2[:, -1] = True                                   
        self.goal_mask_3 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_3[:, -4] = True  
        self.goal_mask_3[:, -1] = True           
        self.goal_mask_4 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_4[:, -3] = True  
        self.goal_mask_4[:, -1] = True           
        self.goal_mask_5 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)     
        self.goal_mask_5[:, -1] = True        
        self.goal_mask_6 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_6[:, -4] = True    
        self.goal_mask_6[:, -3] = True   
        self.goal_mask_6[:, -1] = True           
        self.goal_mask_7 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_7[:, -4] = True    
        self.goal_mask_7[:, -3] = True  
        self.goal_mask_7[:, -2] = True          
        self.goal_mask_8 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)    
        self.goal_mask_8[:, -3] = True   
        self.goal_mask_8[:, -2] = True                                                                 
        self.all_masks = torch.cat([self.goal_mask_0, self.goal_mask_2, self.goal_mask_3, self.goal_mask_5, self.goal_mask_1, self.goal_mask_4, self.goal_mask_6, self.goal_mask_7, self.goal_mask_8], dim=0)
        self.no_mask = torch.zeros((1, self.context_size + 5), dtype=torch.bool) 
        
        avep_mask_0 = (1.0 - self.goal_mask_0.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_0.float())))
        avep_mask_1 = (1.0 - self.goal_mask_1.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_1.float())))
        avep_mask_2 = (1.0 - self.goal_mask_2.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_2.float())))
        avep_mask_3 = (1.0 - self.goal_mask_3.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_3.float())))
        avep_mask_4 = (1.0 - self.goal_mask_4.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_4.float())))
        avep_mask_5 = (1.0 - self.goal_mask_5.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_5.float())))
        avep_mask_6 = (1.0 - self.goal_mask_6.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_6.float())))
        avep_mask_7 = (1.0 - self.goal_mask_7.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_7.float())))
        avep_mask_8 = (1.0 - self.goal_mask_8.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_8.float())))        

        self.avg_pool_mask = torch.cat([avep_mask_0, avep_mask_2, avep_mask_3, avep_mask_5, avep_mask_1, avep_mask_4, avep_mask_6, avep_mask_7, avep_mask_8], dim=0)
        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor, map_images: torch.tensor, goal_img: torch.tensor, goal_mask: torch.tensor, feat_text: torch.tensor, current_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get the goal encoding
        # text feature
        inst_encoding = feat_text
        obsgoal_encoding_lan = self.film_model(current_img, inst_encoding)        
        obsgoal_encoding_lan_cat = obsgoal_encoding_lan.flatten(start_dim=1)
        obsgoal_encoding_lan = self.compress_goal_enc_lan(obsgoal_encoding_lan_cat)  

        if len(obsgoal_encoding_lan.shape) == 2:
            obsgoal_encoding_lan = obsgoal_encoding_lan.unsqueeze(1)
        assert obsgoal_encoding_lan.shape[2] == self.goal_encoding_size
        goal_encoding_lan = obsgoal_encoding_lan   
                
        if self.late_fusion:
            goal_encoding_img = self.goal_encoder_img.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding_img = self.goal_encoder_img.extract_features(obsgoal_img)
        goal_encoding_img = self.goal_encoder_img._avg_pooling(goal_encoding_img)
        if self.goal_encoder._global_params.include_top:
            goal_encoding_img = goal_encoding_img.flatten(start_dim=1)
            goal_encoding_img = self.goal_encoder_img._dropout(goal_encoding_img)
        goal_encoding_img = self.compress_goal_enc_img(goal_encoding_img)

        if len(goal_encoding_img.shape) == 2:
            goal_encoding_img = goal_encoding_img.unsqueeze(1)
        assert goal_encoding_img.shape[2] == self.goal_encoding_size
        
        device = obs_img.get_device()
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        map_encoding = self.goal_encoder.extract_features(map_images).unsqueeze(1)
        map_encoding = self.obs_encoder._avg_pooling(map_encoding)
        
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
            
        if self.goal_encoder._global_params.include_top:
            map_encoding = map_encoding.flatten(start_dim=1)
            map_encoding = self.goal_encoder._dropout(map_encoding)

        obs_encoding = self.compress_obs_enc(obs_encoding)
        map_encoding = self.compress_obs_enc_map(map_encoding)

        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding, map_encoding.unsqueeze(1), goal_encoding_img, goal_encoding_lan), dim=1)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None  
        
        final_repr = self.decoder(tokens, src_key_padding_mask, self.avg_pool_mask.to(device), no_goal_mask)

        action_pred = self.action_predictor(final_repr)
        dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        ) 
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )
              
        return action_pred, dist_pred, no_goal_mask             

def create_conv_layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )

class InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super(InitialFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(
            create_conv_layer(3, 128, 5, 2, 2),
            create_conv_layer(128, 128, 3, 2, 1),
            create_conv_layer(128, 128, 3, 2, 1),
        )
        
    def forward(self, x):
        return self.layers(x)

class IntermediateFeatureExtractor(nn.Module):
    def __init__(self):
        super(IntermediateFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(       
            create_conv_layer(128, 256, 3, 2, 1),
            create_conv_layer(256, 512, 3, 2, 1),
            create_conv_layer(512, 1024, 3, 2, 1),
            create_conv_layer(1024, 1024, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.layers(x)

        
class FiLMTransform(nn.Module):
    def __init__(self):
        super(FiLMTransform, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film_transform = FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film_transform(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x

class FinalClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(FinalClassifier, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        feature_map = x
        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc_layers(x)
        
        return x, feature_map        

class FiLMNetwork(nn.Module):
    def __init__(self, num_res_blocks, num_classes, num_channels, question_dim):
        super(FiLMNetwork, self).__init__()
        question_feature_dim = question_dim

        self.film_param_generator = nn.Linear(question_feature_dim, 2 * num_res_blocks * num_channels)
        self.initial_feature_extractor = InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList()
        self.intermediate_feature_extractor = IntermediateFeatureExtractor()
        
        for _ in range(num_res_blocks):
            self.residual_blocks.append(ResidualBlock(num_channels + 2, num_channels))
            
        self.final_classifier = FinalClassifier(num_channels, num_classes)
    
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        
    def forward(self, x, question):
        batch_size = x.size(0)
        device = x.device
        
        x = self.initial_feature_extractor(x)
        film_params = self.film_param_generator(question).view(
            batch_size, self.num_res_blocks, 2, self.num_channels)
        
        d = x.size(2)
        coords = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).to(device)
        coord_x = coords.expand(batch_size, 1, d, d)
        coord_y = coords.view(d, 1).expand(batch_size, 1, d, d)
        
        for i, res_block in enumerate(self.residual_blocks):
            beta = film_params[:, i, 0, :]
            gamma = film_params[:, i, 1, :]
            
            x = torch.cat([x, coord_x, coord_y], 1)
            x = res_block(x, beta, gamma)
        
        features = self.intermediate_feature_extractor(x)
        
        return features
        
def build_film_model(num_res_blocks, num_classes, num_channels, question_dim):
    return FiLMNetwork(num_res_blocks, num_classes, num_channels, question_dim)                 
