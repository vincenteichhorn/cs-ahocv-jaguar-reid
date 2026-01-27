import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjection(nn.Module):

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=256, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        embeddings = self.network(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ArcFaceLayer(nn.Module):

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1.0, 1.0)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output


class ArcFaceCriterion(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale):
        super().__init__()
        self.arcface = ArcFaceLayer(embedding_dim=embedding_dim, num_classes=num_classes, margin=margin, scale=scale)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.arcface(embeddings, labels)
        loss = self.cross_entropy(logits, labels)
        return loss


class SubCenterArcFaceCriterion(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_subcenters, margin, scale):
        super().__init__()
        self.num_subcenters = num_subcenters
        self.arcface_layers = nn.ModuleList(
            [ArcFaceLayer(embedding_dim=embedding_dim, num_classes=num_classes, margin=margin, scale=scale) for _ in range(num_subcenters)]
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits_list = [arcface_layer(embeddings, labels) for arcface_layer in self.arcface_layers]
        logits = torch.stack(logits_list, dim=1).max(dim=1)[0]
        loss = self.cross_entropy(logits, labels)
        return loss


class TripletCriterion(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        dot_product = torch.matmul(embeddings, embeddings.t())
        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * dot_product, min=1e-16))

        N = labels.size(0)
        mask_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # positive mask, matrix NxN where (i,j) is True if labels[i] == labels[j]
        hardest_pos_dist = (distances * mask_pos.float()).max(1)[0]  # for each sample as anchor, get the hardest positive distance
        hardest_neg_dist = (distances + 1e5 * mask_pos.float()).min(1)[0]  # for each sample as anchor, get the hardest negative distance
        triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)  # for each anchor, compute triplet loss
        loss = triplet_loss.mean()
        return loss


class CosFaceLayer(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale):
        super(CosFaceLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix: (num_classes, embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(norm_embeddings, norm_weight)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = cosine - (one_hot * self.margin)
        output *= self.scale

        return output


class CosFaceCriterion(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale):
        super().__init__()
        self.cosface = CosFaceLayer(embedding_dim=embedding_dim, num_classes=num_classes, margin=margin, scale=scale)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.cosface(embeddings, labels)
        loss = self.cross_entropy(logits, labels)
        return loss


class ArcFaceTripletCriterion(nn.Module):
    def __init__(self, embedding_dim, num_classes, arcface_margin, arcface_scale, triplet_margin, beta=0.5):
        super().__init__()
        self.beta = beta
        self.arcface_criterion = ArcFaceCriterion(embedding_dim, num_classes, arcface_margin, arcface_scale)
        self.triplet_criterion = TripletCriterion(triplet_margin)

    def forward(self, embeddings, labels):
        arcface_loss = self.arcface_criterion(embeddings, labels)
        triplet_loss = self.triplet_criterion(embeddings, labels)
        total_loss = arcface_loss * self.beta + triplet_loss * (1.0 - self.beta)
        return total_loss
