import torch
import torch.nn as nn
import torch.fft
import torch.fft as fft
import numpy as np


class EMWaveNet(nn.Module):
    def __init__(self, num_layers, layer_size, num_classes, wavelength, distance):
        """
        EMWaveNet 초기화.
        - num_layers: 변조층 수.
        - layer_size: 각 층의 크기 (예: 128 for 128x128 이미지).
        - num_classes: 분류할 클래스 수.
        - wavelength: 전자기파의 파장 (λ).
        - distance: 전파 거리 (d).
        """
        super(EMWaveNet, self).__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_classes = num_classes
        self.wavelength = wavelength
        self.distance = distance
        self.k = 2 * np.pi / wavelength  # 파수

        # 학습 가능한 진폭과 위상 매개변수
        self.amplitudes = nn.ParameterList([
            nn.Parameter(torch.ones(layer_size, layer_size)) for _ in range(num_layers)
        ])
        self.phases = nn.ParameterList([
            nn.Parameter(torch.zeros(layer_size, layer_size)) for _ in range(num_layers)
        ])

    def propagate(self, u):
        """전자기파 전파 함수 구현"""
        # 푸리에 변환
        U = fft.fft2(u)
        # 주파수 좌표
        freq_x = fft.fftfreq(u.shape[-2]).reshape(-1, 1)
        freq_y = fft.fftfreq(u.shape[-1]).reshape(1, -1)
        # 전송 함수 (Transfer Function)
        H = torch.exp(1j * self.k * self.distance * (1 -
                      (self.wavelength ** 2 / 2) * (freq_x ** 2 + freq_y ** 2)))
        # 전파 적용
        U_propagated = U * H
        # 역 푸리에 변환
        u_propagated = fft.ifft2(U_propagated)
        return u_propagated

    def forward(self, x):
        """
        순전파 과정.
        - x: 복소수 SAR 이미지 (batch_size, height, width).
        """
        for l in range(self.num_layers):
            # 변조: 진폭과 위상 적용
            modulation = self.amplitudes[l] * \
                torch.exp(1j * self.k * self.phases[l])
            x = x * modulation
            # 전파
            x = self.propagate(x)

        # 출력층: 에너지 계산
        energy = torch.abs(x) ** 2
        return energy


# 예제 사용
if __name__ == "__main__":
    # 임의의 배치: 복소수 형태의 입력 생성 (예: 1채널, 256x256)
    batch_size = 4
    image_size = 256
    # real, imag를 랜덤 생성 후 복소수 텐서로 합성
    x_real = torch.randn(batch_size, 1, image_size, image_size)
    x_imag = torch.randn(batch_size, 1, image_size, image_size)
    x = torch.complex(x_real, x_imag)

    model = EMWaveNet(image_size=image_size, num_layers=5,
                      num_classes=10, wavelength=0.03, d=0.3)
    output = model(x)
    print("출력 로짓 크기:", output.shape)
