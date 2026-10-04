import random

from src.backend import MAX_SEED, GenerationRequest, resolve_job, resolve_seeds


def make_request(**overrides) -> GenerationRequest:
    fields = dict(
        prompt="a {white | black | silver | red} cat",
        negative_prompt="{blurry | lowres}",
        width=512,
        height=512,
        steps=25,
        guidance_scale=7.5,
        sampler="dpmpp_2m_sde_karras",
        seed=-1,
        num_images=1,
    )
    fields.update(overrides)
    return GenerationRequest(**fields)


def test_random_seeds_are_concrete_and_unique():
    seeds = resolve_seeds(-1, 10)
    assert len(seeds) == 10
    assert len(set(seeds)) == 10
    assert all(0 <= s <= MAX_SEED for s in seeds)


def test_random_seeds_unique_even_with_colliding_rng():
    class Colliding(random.Random):
        values = iter([5, 5, 5, 6])

        def randint(self, a, b):
            return next(self.values)

    assert resolve_seeds(-1, 2, Colliding()) == (5, 6)


def test_explicit_seed_sequence():
    assert resolve_seeds(123, 4) == (123, 124, 125, 126)


def test_job_resolves_seeds_and_prompts_per_image():
    job = resolve_job(make_request(seed=123, num_images=4), job_id=7)
    assert job.job_id == 7
    assert job.seeds == (123, 124, 125, 126)
    assert [img.index for img in job.images] == [0, 1, 2, 3]
    for img in job.images:
        assert "{" not in img.prompt and "{" not in img.negative_prompt


def test_same_explicit_job_reproduces_dynamic_choices():
    a = resolve_job(make_request(seed=123, num_images=10))
    b = resolve_job(make_request(seed=123, num_images=10))
    assert a.images == b.images


def test_image_choice_depends_only_on_its_seed():
    # Image with seed 124 resolves the same whether it is first or second in a job.
    second = resolve_job(make_request(seed=123, num_images=2)).images[1]
    first = resolve_job(make_request(seed=124, num_images=1)).images[0]
    assert (second.prompt, second.negative_prompt) == (first.prompt, first.negative_prompt)


def test_random_job_returns_concrete_seeds():
    job = resolve_job(make_request(seed=-1, num_images=10))
    assert -1 not in job.seeds
    assert len(set(job.seeds)) == 10
