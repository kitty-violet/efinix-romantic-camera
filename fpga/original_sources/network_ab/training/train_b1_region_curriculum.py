"""Learn removal of texture before asking for a complete domain conversion.

Only training inputs are mixed. Validation, previews and export always receive
unmodified source frames. A fifth of early crops and every native frame also
receive the true source, and the final fifth of training is source-only.
"""
from pathlib import Path
from . import train_b1_region_formation as parent
from . import train_b1_target_latent as latent
from . import train_b1_late_region_supervision as driver


class CurriculumTraining(parent.FormationTraining):
    def batch_style(self, native):
        source, target, specs, flow, valid = super().batch_style(native)
        progress = self.step / max(self.args.steps, 1)
        amount = .2 + .8 * min(1., max(0., (progress-.2)/.6))
        if native or self.step % 5 == 0:
            amount = 1.
        x = target + amount * (source-target)
        for spec in specs:
            spec['training_source_fraction'] = amount
        return x, target, specs, flow, valid


if __name__ == '__main__':
    parent.unit_check()
    driver.main(training_factory=CurriculumTraining,
                extra_sources=[Path(__file__), Path(parent.__file__), Path(latent.__file__)],
                evaluation=parent.evaluate)
