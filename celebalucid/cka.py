import numpy as np
from tqdm import tqdm
import torch
import gc
import warnings

from celebalucid.generator import build_generator
from celebalucid import load_model


class CKA:

    def __init__(self, workdir, dataset_name, model_names, n_epochs=10, n_iters=64, batch_size=32):
        self.workdir = workdir
        self.dataset_name = dataset_name
        self.set_models(model_names)
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.batch_size = batch_size

        self._get_gen()  # Make sure data is downloaded
        self._revise_n_iters()

    def set_models(self, model_names):
        if hasattr(self, 'models'):
            self.models[0].switch_to(model_names[0])
            self.models[1].switch_to(model_names[1])
        else:
            self.models = [load_model(name) for name in model_names]

    def __call__(self, layer, verbose=True):
        # CKA results
        similarities = []

        for epoch_i in self._verbose_range(self.n_epochs, verbose):

            # Save list of activations
            models_results = [[], []]

            for model_i, model in enumerate(self.models):

                # Generator uniform sampling images
                gen_iterator = iter(self._get_gen(seed=epoch_i))

                for _ in range(self.n_iters):

                    # Run network and save activations
                    imgs, _ = next(gen_iterator)
                    activations = self._get_activations(model, layer, imgs)
                    activations = list(activations.reshape(len(imgs), -1))
                    models_results[model_i].extend(activations)

                models_results[model_i] = np.array(models_results[model_i])

            # Calculate CKA
            x1 = self.gram_linear(models_results[0])
            x2 = self.gram_linear(models_results[1])
            similarity = self.cka(x1, x2)
            
            # Save
            similarities.append(similarity)

        # Return with mean CKA over epochs
        mean_similarity = np.mean(similarities)
        return mean_similarity

    def _get_gen(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return build_generator(self.workdir, self.dataset_name,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=1, pin_memory=False,
                               verbose=False, drop_last=True)

    def _revise_n_iters(self):
        n_max_batches = len(self._get_gen())
        if self.n_iters > n_max_batches:
            warnings.warn('Number of iterations (n_iters) is larger then '
                          'number of available batches in generator. Therefore '
                          'n_iters made equal to size of the generator.')
            self.n_iters = n_max_batches

    def gram_linear(self, x):
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        return x.dot(x.T)

    def center_gram(self, gram, unbiased=False):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
            A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(self, gram_x, gram_y, debiased=False):
        """Compute CKA.

        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
            The value of CKA between X and Y.
        """
        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def _get_activations(self, model, layer, imgs):
        imgs = imgs.to(model.device)
        model.stream(imgs)
        activations = model.neurons[layer].detach().cpu().numpy()
        return activations

    def _verbose_range(self, n, verbose):
        range_epochs = range(n)
        range_epochs = tqdm(range_epochs) if verbose else range_epochs
        return range_epochs
