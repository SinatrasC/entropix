import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import tyro
import numpy as np
import pygame

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample, calculate_metrics, SamplerState
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def apply_scaling(freqs: jax.Array):
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask


def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')):
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption('Adaptive Sampling Visualization')
    screen_width, screen_height = 1400, 1100  # Increased height to accommodate new charts
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Font settings
    font_size = 24
    font = pygame.font.SysFont('monospace', font_size)
    line_height = font_size + 4

    # Initialize variables for plotting
    entropies = []
    varentropies = []
    steps = []
    sampler_states = []  # New list for sampler states

    # Set up the plotting area dimensions
    margin = 10
    legend_height = 70
    xy_plot_height = 200
    line_plot_height = 200
    sampler_state_plot_height = 100  # New height for sampler state chart

    text_area_start_y = legend_height + xy_plot_height + line_plot_height + sampler_state_plot_height + 6 * margin

    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights(weights_path.absolute())
    tokenizer = Tokenizer('entropix/tokenizer.model')

    # Create the batch of tokens
    def generate(xfmr_weights, model_params, tokens):
        gen_tokens = tokens
        cur_pos = tokens.shape[1]
        tokens = jnp.array(tokens, jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, 0)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)

        # Get initial logits and attention scores
        logits, kvcache, attention_scores, _ = xfmr(xfmr_weights, model_params, tokens, 0, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)

        # Define stop tokens
        stop = jnp.array([128001, 128008, 128009])
        sampler_cfg = SamplerConfig()

        # Initialize variables for text display
        max_text_width = screen_width - 20

        # Initialize a buffer to store all rendered tokens with their colors
        rendered_tokens = []  # List of tuples: (decoded_text, color)

        # Clear the screen initially
        screen.fill((255, 255, 255))
        pygame.display.flip()

        running = True
        next_token = None
        key = jax.random.PRNGKey(1337)  # Initialize the random key

        while running and cur_pos < 8192:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

            if next_token is not None:
                # Generate logits for the next token
                logits, kvcache, attention_scores, _ = xfmr(
                    xfmr_weights,
                    model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos:cur_pos+1],
                    kvcache
                )
                gen_tokens = jnp.concatenate((gen_tokens, next_token), axis=1)

            # Calculate metrics
            metrics = calculate_metrics(logits, attention_scores)
            entropy = metrics['logits_entropy'].item()
            varentropy = metrics['logits_varentropy'].item()
            attn_entropy = metrics['attn_entropy'].item()
            attn_varentropy = metrics['attn_varentropy'].item()
            agreement = metrics['agreement'].item()
            interaction_strength = metrics['interaction_strength'].item()

            # Update metrics lists
            entropies.append(entropy)
            varentropies.append(varentropy)
            steps.append(cur_pos)

            # Compute dynamic maximums
            MAX_ENTROPY = max(entropies) if entropies else 10.0
            MAX_VARENTROPY = max(varentropies) if varentropies else 10.0

            # Sample the next token and get sampler state
            next_token, sampler_state = sample(gen_tokens, logits, attention_scores, cfg=sampler_cfg, key=key)
            sampler_states.append(sampler_state)

            # Determine if the current state is ADAPTIVE
            is_adaptive = (sampler_state == SamplerState.ADAPTIVE)

            # Get color based on entropy and varentropy, adjusted for adaptive state
            color = color_for_metrics(entropy, varentropy, MAX_ENTROPY, MAX_VARENTROPY, is_adaptive=is_adaptive)

            # Decode the token
            decoded_text = tokenizer.decode(next_token.tolist()[0])

            # Append decoded text and its color to the rendered_tokens list
            rendered_tokens.append((decoded_text, color))

            # **Update plots first**
            plot_entropy_varentropy_plane(
                screen,
                entropy,
                varentropy,
                screen_width,
                xy_plot_height,
                legend_height + margin,
                MAX_ENTROPY,
                MAX_VARENTROPY
            )
            plot_metrics(
                screen,
                entropies,
                varentropies,
                steps,
                screen_width,
                line_plot_height,
                legend_height + xy_plot_height + 2 * margin
            )
            plot_sampler_state(
                screen,
                sampler_states,
                steps,
                screen_width,
                sampler_state_plot_height,
                legend_height + xy_plot_height + line_plot_height + 3 * margin
            )

            draw_legend(screen, screen_width, 0)

            # **Render the text after plotting**
            # Clear the text area
            text_area_rect = pygame.Rect(
                0,
                text_area_start_y,
                screen_width,
                screen_height - text_area_start_y
            )
            pygame.draw.rect(screen, (255, 255, 255), text_area_rect)

            # Initialize text position
            text_x, text_y = 10, text_area_start_y + 10
            max_text_width = screen_width - 20

            # Render all tokens with their respective colors
            for token_text, token_color in rendered_tokens:
                # Render the token
                text_surface = font.render(token_text, True, token_color)
                text_width = text_surface.get_width()

                # Check for wrapping
                if text_x + text_width > max_text_width:
                    text_x = 10
                    text_y += line_height

                # Blit the token
                screen.blit(text_surface, (text_x, text_y))
                text_x += text_width

            # Update current position
            cur_pos += 1

            # Check for stop tokens
            if jnp.isin(next_token, stop).any():
                running = False

            # Refresh the display
            pygame.display.flip()

        # Keep the window open after generation is complete
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


    def color_for_metrics(entropy_value, varentropy_value, max_entropy, max_varentropy, is_adaptive=False):
        """
        Maps entropy and varentropy values to RGB colors with logarithmic scaling.
        When is_adaptive is True, the color is blended with gray to be less vibrant.

        Args:
            entropy_value (float): The entropy metric value.
            varentropy_value (float): The varentropy metric value.
            max_entropy (float): The current maximum entropy value.
            max_varentropy (float): The current maximum varentropy value.
            is_adaptive (bool): Flag indicating if the sampler is in ADAPTIVE state.

        Returns:
            tuple: A tuple representing the RGB color.
        """
        # Normalize the entropy and varentropy values to a range between 0 and 1
        entropy_norm = min(max(entropy_value / max_entropy, 0.0), 1.0)
        varentropy_norm = min(max(varentropy_value / max_varentropy, 0.0), 1.0)

        # Apply logarithmic scaling
        # Adding a small epsilon to avoid log(0)
        epsilon = 1e-6
        entropy_log = math.log1p(entropy_norm * 9 + epsilon) / math.log(10)  # log1p(x) = log(1 + x)
        varentropy_log = math.log1p(varentropy_norm * 9 + epsilon) / math.log(10)

        # Compute color components with log scaling
        # Red increases with varentropy, Blue increases with entropy, Green decreases with varentropy
        red_intensity = int(varentropy_log * 255)
        blue_intensity = int(entropy_log * 255)
        green_intensity = int((1 - varentropy_log) * 255)

        # Ensure color values are within [0, 255]
        red_intensity = min(max(red_intensity, 0), 255)
        blue_intensity = min(max(blue_intensity, 0), 255)
        green_intensity = min(max(green_intensity, 0), 255)

        color = (red_intensity, green_intensity, blue_intensity)

        if is_adaptive:
            # Blend the color with gray (128, 128, 128) to make it less vibrant
            gray = (128, 128, 128)
            blend_factor = 0.7  # Adjust the blend factor as needed (0 = original color, 1 = gray)
            blended_color = tuple(int((1 - blend_factor) * c + blend_factor * g) for c, g in zip(color, gray))
            return blended_color

        return color

    def plot_entropy_varentropy_plane(screen, entropy_value, varentropy_value, plot_width, plot_height, plot_y_start, max_entropy, max_varentropy):
        # Clear the area
        pygame.draw.rect(screen, (255, 255, 255), (0, plot_y_start, plot_width, plot_height))

        # Draw axes
        axis_color = (0, 0, 0)  # black
        pygame.draw.line(screen, axis_color, (plot_width // 2, plot_y_start), (plot_width // 2, plot_y_start + plot_height), 2)
        pygame.draw.line(screen, axis_color, (0, plot_y_start + plot_height // 2), (plot_width, plot_y_start + plot_height // 2), 2)

        # Determine the position of the point
        # Normalize based on dynamic maximums
        entropy_norm = (entropy_value / max_entropy) * 2 - 1
        varentropy_norm = (varentropy_value / max_varentropy) * 2 - 1

        # Compute position on the plot
        x_pos = (entropy_norm + 1) / 2 * plot_width
        y_pos = (1 - (varentropy_norm + 1) / 2) * plot_height + plot_y_start  # Invert y-axis for Pygame

        # Use dynamic color based on metrics
        point_color = color_for_metrics(entropy_value, varentropy_value, max_entropy, max_varentropy)

        # Draw the point with the dynamic color
        pygame.draw.circle(screen, point_color, (int(x_pos), int(y_pos)), 5)

        # Add labels
        font_small = pygame.font.SysFont('monospace', 16)
        entropy_label = font_small.render('Entropy Low → High', True, axis_color)
        varentropy_label = font_small.render('Varentropy Low ↑ High', True, axis_color)
        screen.blit(entropy_label, (plot_width // 2 + 5, plot_y_start + 5))
        screen.blit(varentropy_label, (5, plot_y_start + plot_height // 2 + 5))


    def plot_metrics(screen, entropies, varentropies, steps, plot_width, plot_height, plot_y_start):
        # Clear the plotting area
        pygame.draw.rect(screen, (255, 255, 255), (0, plot_y_start, plot_width, plot_height + margin))

        if len(steps) > 1:
            # Scale metrics to fit the plot area
            max_entropy = max(entropies) if entropies else 10.0
            max_varentropy = max(varentropies) if varentropies else 10.0

            # Avoid division by zero
            max_entropy = max_entropy if max_entropy != 0 else 1
            max_varentropy = max_varentropy if max_varentropy != 0 else 1

            entropy_points = [
                (margin + (step - steps[0]) / (steps[-1] - steps[0] + 1) * (plot_width - 2 * margin),
                 plot_y_start + plot_height - (entropy / max_entropy) * (plot_height - margin))
                for step, entropy in zip(steps, entropies)
            ]

            varentropy_points = [
                (margin + (step - steps[0]) / (steps[-1] - steps[0] + 1) * (plot_width - 2 * margin),
                 plot_y_start + plot_height - (varentropy / max_varentropy) * (plot_height - margin))
                for step, varentropy in zip(steps, varentropies)
            ]

            # Draw entropy line (blue)
            if len(entropy_points) > 1:
                pygame.draw.lines(screen, (0, 0, 255), False, entropy_points, 2)

            # Draw varentropy line (red)
            if len(varentropy_points) > 1:
                pygame.draw.lines(screen, (255, 0, 0), False, varentropy_points, 2)

            # Labels
            font_small = pygame.font.SysFont('monospace', 16)
            entropy_label = font_small.render('Entropy', True, (0, 0, 255))
            varentropy_label = font_small.render('Varentropy', True, (255, 0, 0))
            screen.blit(entropy_label, (10, plot_y_start + 10))
            screen.blit(varentropy_label, (10, plot_y_start + 30))


    def plot_sampler_state(screen, sampler_states, steps, plot_width, plot_height, plot_y_start):
        # Clear the plotting area
        pygame.draw.rect(screen, (255, 255, 255), (0, plot_y_start, plot_width, plot_height))

        if len(steps) > 1 and len(sampler_states) > 1:
            # Define colors for each sampler state
            state_colors = {
                SamplerState.FLOWING: (0, 255, 0),          # Green
                SamplerState.TREEDING: (255, 255, 0),      # Yellow
                SamplerState.EXPLORING: (255, 165, 0),     # Orange
                SamplerState.RESAMPLING: (255, 0, 0),      # Red
                SamplerState.ADAPTIVE: (0, 0, 255),        # Blue
            }

            # Scale steps to fit the plot width
            step_range = steps[-1] - steps[0] + 1
            x_scale = (plot_width - 2 * margin) / step_range

            for idx, step in enumerate(steps):
                x = margin + (step - steps[0]) * x_scale
                y = plot_y_start + plot_height // 2
                state = sampler_states[idx]
                color = state_colors.get(state, (128, 128, 128))  # Default to gray if state not found
                pygame.draw.circle(screen, color, (int(x), int(y)), 3)

        # Add labels
        font_small = pygame.font.SysFont('monospace', 16)
        label = font_small.render('Sampler State', True, (0, 0, 0))
        screen.blit(label, (10, plot_y_start + 10))


    def draw_legend(screen, plot_width, plot_y_start):
        # Clear the area
        pygame.draw.rect(screen, (255, 255, 255), (0, plot_y_start, plot_width, legend_height))

        font_small = pygame.font.SysFont('monospace', 16)

        # Define colors and labels
        states = [
            ('Flowing with unspoken intent', (0, 255, 0)),          # Green
            ('Treading carefully, asking clarifying questions', (255, 255, 0)),  # Yellow
            ('Exploring forks in the path', (255, 165, 0)),         # Orange
            ('Resampling in the mist', (255, 0, 0)),                # Red
            ('Adaptive Sampling', (0, 0, 255)),                     # Blue
        ]

        x = 10
        y = plot_y_start + 10
        for label, color in states:
            rect = pygame.Rect(x, y, 20, 20)
            pygame.draw.rect(screen, color, rect)
            text_surface = font_small.render(label, True, (0, 0, 0))
            screen.blit(text_surface, (x + 25, y))
            y += 25


    csv_path = Path('entropix/data/prompts.csv')
    prompts = create_prompts_from_csv(csv_path)
    PROMPT_TEST = False

    if PROMPT_TEST:
        for p in prompts:
            print(p)
            tokens = tokenizer.encode(p, bos=False, eos=False, allowed_special='all')
            tokens = jnp.array([tokens])  # Adjusted for batch dimension
            generate(xfmr_weights, model_params, tokens)
    else:
        print(prompt)
        tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        tokens = jnp.array([tokens])  # Adjusted for batch dimension
        generate(xfmr_weights, model_params, tokens)


def color_for_metrics(entropy_value, varentropy_value, max_entropy, max_varentropy):
    """
    Maps entropy and varentropy values to RGB colors with logarithmic scaling.

    Args:
        entropy_value (float): The entropy metric value.
        varentropy_value (float): The varentropy metric value.
        max_entropy (float): The current maximum entropy value.
        max_varentropy (float): The current maximum varentropy value.

    Returns:
        tuple: A tuple representing the RGB color.
    """
    # Normalize the entropy and varentropy values to a range between 0 and 1
    entropy_norm = min(max(entropy_value / max_entropy, 0.0), 1.0)
    varentropy_norm = min(max(varentropy_value / max_varentropy, 0.0), 1.0)

    # Apply logarithmic scaling
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-6
    entropy_log = math.log1p(entropy_norm * 9 + epsilon) / math.log(10)  # log1p(x) = log(1 + x)
    varentropy_log = math.log1p(varentropy_norm * 9 + epsilon) / math.log(10)

    # Compute color components with log scaling
    # Red increases with varentropy, Blue increases with entropy, Green decreases with varentropy
    red_intensity = int(varentropy_log * 255)
    blue_intensity = int(entropy_log * 255)
    green_intensity = int((1 - varentropy_log) * 255)

    # Ensure color values are within [0, 255]
    red_intensity = min(max(red_intensity, 0), 255)
    blue_intensity = min(max(blue_intensity, 0), 255)
    green_intensity = min(max(green_intensity, 0), 255)

    return (red_intensity, green_intensity, blue_intensity)


if __name__ == '__main__':
    tyro.cli(main)
