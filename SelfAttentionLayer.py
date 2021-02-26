class AttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, weight_dim, device):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))

        self.feature_dim = weight_dim

    def forward(self, x):
        '''
        inference
        :param x: [num_agent, num_target, feature_dim]
        :return z: [num_agent, num_target, weight_dim]
        '''
        # z = softmax(Q,K)*V
        q = torch.tanh(self.Q(x))  # [batch_size, sequence_len, weight_dim]
        k = torch.tanh(self.K(x))  # [batch_size, sequence_len, weight_dim]
        v = torch.tanh(self.V(x))  # [batch_size, sequence_len, weight_dim]

        z = torch.bmm(F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=2), v)  # [batch_size, sequence_len, weight_dim]

        global_feature = z.sum(dim=1)
        return z, global_feature
