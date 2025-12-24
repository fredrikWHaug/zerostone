pub struct OnlineStats<const C: usize> {
    count: u64,
    mean: [f64; C],
    m2: [f64; C],
}

impl<const C: usize> OnlineStats<C> {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: [0.0; C],
            m2: [0.0; C],
        }
    }

    pub fn update(&mut self, sample: &[f64; C]) {
        self.count += 1;
        let n = self.count as f64;

        for i in 0..C {
            let delta = sample[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = sample[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    pub fn mean(&self) -> &[f64; C] {
        &self.mean
    }

    pub fn variance(&self) -> [f64; C] {
        if self.count < 2 {
            return [0.0; C];
        }

        let mut var = [0.0; C];
        for i in 0..C {
            var[i] = self.m2[i] / (self.count - 1) as f64;
        }
        var
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = [0.0; C];
        self.m2 = [0.0; C];
    }
}
