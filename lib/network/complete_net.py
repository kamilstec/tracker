import torch.nn as nn
from .optimizationGNN import optimNet
from .encoderCNN import EncoderCNNdensenet, EncoderCNNresnet, EncoderCNNefficientnet
from .affinity import affinityNet
from .affinity_appearance import affinity_appearanceNet
from .affinity_geom import affinity_geomNet
from .utils import box_iou_calc, sinkhorn
import torch
from .affinity_final import affinity_finalNet

class completeNet(nn.Module):
    def __init__(self, cnn_encoder):
        super(completeNet, self).__init__()
        if cnn_encoder == 'efficientnet-b3' or cnn_encoder == 'efficientnet-b0':
            self.cnn = EncoderCNNefficientnet(cnn_encoder)
        elif cnn_encoder == 'densenet121':
            self.cnn = EncoderCNNdensenet()
        elif cnn_encoder == 'resnet18' or cnn_encoder == 'resnet34' or cnn_encoder == 'resnet50' or cnn_encoder == \
                'resnet101':
            self.cnn = EncoderCNNresnet(cnn_encoder)
        else:
            print('Co to za sieć?')
        self.affinity_appearance_net = affinity_appearanceNet(cnn_encoder)
        self.optim_net = optimNet(cnn_encoder)  # faktyczna sieć grafowa
        self.affinity_geom_net = affinity_geomNet()
        self.affinity_net = affinityNet()
        self.affinity_final_net = affinity_finalNet()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, data):
        x, coords_original, edge_index, ground_truth, coords, track_num, detections_num = \
            data.x, data.coords_original, data.edge_index, data.ground_truth, data.coords_normalized, data.track_num, data.det_num

        if torch.cuda.is_available():
            slack = torch.Tensor([-0.2]).float().cuda()
            lam = torch.Tensor([5]).float().cuda()
        else:
            slack = torch.Tensor([-0.2]).float()
            lam = torch.Tensor([5]).float()

        # Pass through GNN
        node_embedding = self.cnn(x)
        edge_embedding = []
        edge_mlp = []
        if len(edge_index[0]) == 0:
            pass
        for i in range(len(edge_index[0])):
            x1 = self.affinity_appearance_net(torch.cat((node_embedding[edge_index[0][i]], node_embedding[edge_index[1][i]]), 0))
            x2 = self.affinity_geom_net(torch.cat((coords[edge_index[0][i]], coords[edge_index[1][i]]), 0))
            iou= box_iou_calc(coords_original[edge_index[0][i]], coords_original[edge_index[1][i]])
            edge_mlp.append(iou)
            # Pass through mlp
            inputs = torch.cat((x1.reshape(1), x2.reshape(1)), 0)
            edge_embedding.append(self.affinity_net(inputs))
        edge_embedding = torch.stack(edge_embedding)
        output = self.optim_net(node_embedding, edge_embedding, edge_index)
        output_temp = []
        for i in range(len(edge_index[0])):
            if edge_index[0][i]<edge_index[1][i]:
                nodes_difference= self.cos(output[edge_index[0][i]], output[edge_index[1][i]])
                x1 = self.affinity_final_net(torch.cat((nodes_difference.reshape(1), edge_mlp[i].reshape(1)), 0))
                output_temp.append(x1.reshape(1))
        output = output_temp
        start1= 0
        start2 = 0
        normalized_output= []
        tracklet_num = []
        det_num = []
        for i,j in enumerate([1]):
            num_of_edges1= data.num_edges
            num_of_edges2= int(num_of_edges1/2)
            output_sliced = output
            edges_sliced = edge_index
            start1 += num_of_edges1
            start2 += num_of_edges2

            row, col = edges_sliced
            mask = row < col
            edges_sliced = edges_sliced[:, mask]
            matrix = []
            for k in range(int(track_num[i].item())):
                matrix.append([])
                for l in range(int(detections_num[i].item())):
                    if torch.cuda.is_available():
                        matrix[k].append(torch.zeros(1, dtype=torch.float, requires_grad=False).cuda())
                    else:
                        matrix[k].append(torch.zeros(1, dtype=torch.float, requires_grad=False))
                matrix[k].append(torch.exp(slack*lam))#slack
            for k,m in enumerate(edges_sliced[0]):
                matrix[int(edges_sliced[0,k].item())][int(edges_sliced[1,k].item())-int(track_num[i].item())]=torch.exp(output_sliced[k]*lam)
            for w,z in enumerate(matrix):
                matrix[w] = torch.cat(z)
            if torch.cuda.is_available():
                matrix.append(torch.ones(len(matrix[0])).cuda()*torch.exp(slack*lam))#slack
            else:
                matrix.append(torch.ones(len(matrix[0])) * torch.exp(slack * lam))  # slack
            matrix = torch.stack(matrix)
            matrix = sinkhorn(matrix)
            matrix = matrix[0:-1,0:-1]
            if torch.cuda.is_available():
                det_num.append(torch.tensor(len(matrix[0]), dtype= int).cuda())
                tracklet_num.append(torch.tensor(len(matrix), dtype= int).cuda())
            else:
                det_num.append(torch.tensor(len(matrix[0]), dtype=int))
                tracklet_num.append(torch.tensor(len(matrix), dtype=int))
            normalized_output.append(matrix.reshape(-1))
        normalized_output = torch.cat((normalized_output[:]),dim=0)
        normalized_output_final= []
        ground_truth_final= []
        for k, l in enumerate(normalized_output):
            if l.item()!=0:
                normalized_output_final.append(l)
                ground_truth_final.append(ground_truth[k])
        return torch.stack(normalized_output_final), normalized_output, torch.stack(ground_truth_final), ground_truth, torch.stack(det_num), torch.stack(tracklet_num)
