class Losses:
    ORTHOGONALITY = "orth"
    CONSISTENCY = "cons"
    DIFFUSION = "diff"
    BPR = "bpr"

    @staticmethod
    def validate(losses):
        valid_losses = {
            Losses.ORTHOGONALITY,
            Losses.CONSISTENCY,
            Losses.DIFFUSION,
            Losses.BPR
        }

        assert losses and all([loss in valid_losses for loss in losses]), "none or invalid layers"
