import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        # Calculate log probabilities
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Gather the log probabilities of the target classes
        log_p_t = log_p.gather(1, labels.view(-1, 1))
        p_t = p.gather(1, labels.view(-1, 1))

        # Focal Loss formula
        loss = -((1 - p_t) ** self.gamma) * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalArcFaceCriterion(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale, gamma=2.0):
        super().__init__()
        self.arcface = ArcFaceLayer(embedding_dim=embedding_dim, num_classes=num_classes, margin=margin, scale=scale)
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, embeddings, labels):
        logits = self.arcface(embeddings, labels)
        loss = self.focal_loss(logits, labels)
        return loss


class HyperbolicArcFaceCriterion(nn.Module):
    """
    ArcFace-style classification loss for embeddings living in the
    Poincaré ball.  Instead of using cosine similarity (natural on the
    hypersphere), this computes similarity from the negative Poincaré
    distance between embeddings and learnable class prototypes that
    also live in the ball.
    """

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0, curvature=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.curvature = curvature

        # Class prototypes in tangent space (mapped to ball at forward time)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.cross_entropy = nn.CrossEntropyLoss()

    def _exp_map_zero(self, v):
        sqrt_c = self.curvature**0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def _poincare_distance(self, x, y):
        """Poincaré distance between points x and y in the ball."""
        c = self.curvature
        diff = x - y
        diff_norm_sq = diff.pow(2).sum(dim=-1)
        x_norm_sq = x.pow(2).sum(dim=-1).clamp(max=1 - 1e-5)
        y_norm_sq = y.pow(2).sum(dim=-1).clamp(max=1 - 1e-5)
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        arg = 1 + 2 * c * diff_norm_sq / denom.clamp(min=1e-7)
        return torch.acosh(arg.clamp(min=1.0 + 1e-7)) / (c**0.5)

    def forward(self, embeddings, labels):
        # Map class prototypes into the Poincaré ball
        prototypes = self._exp_map_zero(self.weight)
        proto_norm = prototypes.norm(dim=-1, keepdim=True)
        prototypes = torch.where(proto_norm > 0.95, prototypes * 0.95 / proto_norm, prototypes)

        # Compute pairwise Poincaré distances → convert to similarity
        # embeddings: (B, D), prototypes: (C, D) → distances: (B, C)
        dists = torch.stack([self._poincare_distance(embeddings, prototypes[j].unsqueeze(0)) for j in range(self.num_classes)], dim=1)

        # Convert distances to cosine-like similarities for ArcFace margin
        # Using exp(-d) as similarity, then normalizing to [-1, 1] range
        sims = torch.exp(-dists)
        # Normalize to approximate cosine range for margin arithmetic
        cosine = 2 * sims - 1
        cosine = cosine.clamp(-1.0, 1.0)

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.cross_entropy(output, labels)
        return loss
