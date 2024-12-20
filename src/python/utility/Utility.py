class Utility:
    """
        This update target parameters slowly
        Based on rate `tau`, which is much less than one.
    """
    @staticmethod
    def update_target(target, original, tau):
        target_weights = target.get_weights()
        original_weights = original.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

        target.set_weights(target_weights)