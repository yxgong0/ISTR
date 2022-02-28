import torch
import torch.nn as nn


class SampleLoss(nn.Module):

    def __init__(self, colored=False, num_clusters=25, min_pixel=0, max_pixel=255):
        super(SampleLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.colored = colored
        self.num_clusters = num_clusters
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def forward(self, _input, _target):
        assert _input.shape == _target.shape

        h = _input.shape[2]
        w = _input.shape[3]
        pixels = h * w

        _input = self.sort_tensor(_input)
        _target = self.sort_tensor(_target)

        pixel_area = self.max_pixel - self.min_pixel
        step = pixel_area / self.num_clusters
        nodes = [self.min_pixel + step * n for n in range(self.num_clusters)]

        loss_sum = 0

        for i in range(nodes.__len__() - 1):
            # min_tensor = torch.FloatTensor(_input.shape).fill_(nodes[i])
            # max_tensor = torch.FloatTensor(_input.shape).fill_(nodes[i+1])
            zeros = torch.zeros_like(_input)
            ones = torch.ones_like(_input)

            input_mask = self.sort_tensor(torch.where((torch.where(_input >= nodes[i], ones, zeros)
                                                       + torch.where(_input < nodes[i + 1], ones, zeros)) > 1.001,
                                                      ones, zeros))
            target_mask = self.sort_tensor(torch.where((torch.where(_target >= nodes[i], ones, zeros)
                                                       + torch.where(_target < nodes[i + 1], ones, zeros)) > 1.001,
                                                       ones, zeros))

            diff1 = torch.where(input_mask + target_mask >= 1.001, zeros, ones)
            diff2 = torch.where(input_mask + target_mask <= 0.999, zeros, ones)
            diff = self.sort_tensor(torch.where(diff1 + diff2 >= 1.001, ones, zeros))

            diff_num = torch.sum(diff)
            input_num = torch.sum(input_mask)
            target_sum = torch.sum(target_mask)

            loss_ = torch.clamp(diff_num - 0.5 * min(input_num, target_sum), 0, pixels) / pixels
            loss_sum += loss_

        loss = loss_sum / (nodes.__len__() - 1)

        return loss

    def sort_tensor(self, tensor):
        batch_size = tensor.shape[0]
        nc = tensor.shape[1]
        if tensor.shape.__len__() == 4:
            h = tensor.shape[2]
            w = tensor.shape[3]

            tensor = tensor.view(batch_size, nc, h*w)

        c1 = tensor[:, 0, :]
        if self.colored:
            c2 = tensor[:, 1, :]
            c3 = tensor[:, 2, :]

        c1_sorted, indices = torch.sort(c1, 1)
        _, indices = torch.sort(indices)

        if self.colored:
            c2_sorted = self.sort_from_indices(c2, indices)
            c3_sorted = self.sort_from_indices(c3, indices)

        tensor_sorted = c1_sorted.unsqueeze(1)
        if self.colored:
            tensor_sorted = torch.cat((tensor_sorted, c2_sorted.unsqueeze(1)), 1)
            tensor_sorted = torch.cat((tensor_sorted, c3_sorted.unsqueeze(1)), 1)

        return tensor_sorted

    @staticmethod
    def sort_from_indices(tensor, indices):
        tensor_sorted = torch.Tensor()
        i = 0
        for img_vector in tensor:
            img_vector_index = indices[i]
            i += 1
            img_vector_sorted = torch.index_select(img_vector, 0, img_vector_index)
            if i == 1:
                tensor_sorted = img_vector_sorted
                tensor_sorted = tensor_sorted.unsqueeze(0)
            else:
                img_vector_sorted = img_vector_sorted.unsqueeze(0)
                tensor_sorted = torch.cat((tensor_sorted, img_vector_sorted), 0)

        return tensor_sorted


if __name__ == '__main__':
    a = torch.Tensor([[[[15, 25, 25], [0, 15, 25], [0, 0, 15]]]])
    b = torch.Tensor([[[[0, 0, 0], [15, 15, 15], [0, 0, 0]]]])

    criterion = SampleLoss()
    loss = criterion(a, b)
