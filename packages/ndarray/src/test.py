import torch as th

a = th.tensor(
    [
        [
            [
                0,
                1,
            ],
        
            [
                2,
                3,
            ],
        ],
        [
            [
                4,
                5,
            ],
            [
                6,
                7,
            ],
        ],
    ]
)

print(a.shape)