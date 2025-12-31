pub struct OnlineStats<const C: usize> {
    count: u64,
    mean: [f64; C],
    m2: [f64; C],
}

impl<const C: usize> Default for OnlineStats<C> {
    fn default() -> Self {
        Self::new()
    }
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

        for (i, &s) in sample.iter().enumerate() {
            let delta = s - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = s - self.mean[i];
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
        for (v, &m) in var.iter_mut().zip(self.m2.iter()) {
            *v = m / (self.count - 1) as f64;
        }
        var
    }

    /// Returns the standard deviation for each dimension.
    ///
    /// This is the square root of the sample variance.
    pub fn std_dev(&self) -> [f64; C] {
        let var = self.variance();
        let mut std = [0.0; C];
        for (s, &v) in std.iter_mut().zip(var.iter()) {
            *s = libm::sqrt(v);
        }
        std
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
