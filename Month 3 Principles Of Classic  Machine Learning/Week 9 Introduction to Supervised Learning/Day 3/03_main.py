
#Part 3: Manim Animations for Regression Concepts

#MANIM ANIMATIONS FOR POLYNOMIAL REGRESSION
#Visualize regression concepts with animations
#To run these animations, you need Manim installed:
#Where SceneName is one of:
#- PolynomialFitScene
#- BiasVarianceScene
#- RegularizationScene
#- GradientDescentScene


from manim import *
import numpy as np

class PolynomialFitScene(Scene):
    """Animate polynomial fitting with different degrees"""
    
    def construct(self):
        # Title
        title = Text("Polynomial Regression", font_size=48, weight=BOLD)
        subtitle = Text("Fitting curves to data", font_size=28)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-10, 10, 5],
            axis_config={"color": BLUE},
            x_length=10,
            y_length=6
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y", edge=LEFT, direction=LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generate data points
        np.random.seed(42)
        x_data = np.linspace(-2.5, 2.5, 30)
        y_true = 0.5 * x_data**3 - 2 * x_data**2 + x_data + 1
        y_noisy = y_true + np.random.normal(0, 1, len(x_data))
        
        # Create dots for data points
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=YELLOW, radius=0.06)
            for x, y in zip(x_data, y_noisy)
        ])
        
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.05))
        self.wait()
        
        # Fit polynomials of different degrees
        degrees = [1, 2, 3, 10]
        colors = [RED, GREEN, BLUE, PURPLE]
        
        for degree, color in zip(degrees, colors):
            # Fit polynomial
            coeffs = np.polyfit(x_data, y_noisy, degree)
            poly_func = np.poly1d(coeffs)
            
            # Create polynomial curve
            curve = axes.plot(
                lambda x: poly_func(x),
                color=color,
                x_range=[-2.5, 2.5]
            )
            
            # Label
            label = Text(f"Degree {degree}", font_size=28, color=color)
            label.to_corner(UR).shift(DOWN * (degrees.index(degree) * 0.6))
            
            self.play(Create(curve), Write(label))
            self.wait(1)
            
            if degree < degrees[-1]:
                self.play(FadeOut(curve), FadeOut(label))
        
        self.wait(2)


class BiasVarianceScene(Scene):
    """Visualize bias-variance tradeoff"""
    
    def construct(self):
        # Title
        title = Text("Bias-Variance Tradeoff", font_size=48, weight=BOLD)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))
        
        # Create three scenarios
        scenarios = VGroup()
        
        # Underfitting
        under_title = Text("Underfitting", font_size=28, color=RED)
        under_subtitle = Text("High Bias", font_size=20, color=RED)
        under_subtitle.next_to(under_title, DOWN)
        under_group = VGroup(under_title, under_subtitle)
        
        # Just Right
        optimal_title = Text("Optimal", font_size=28, color=GREEN)
        optimal_subtitle = Text("Balanced", font_size=20, color=GREEN)
        optimal_subtitle.next_to(optimal_title, DOWN)
        optimal_group = VGroup(optimal_title, optimal_subtitle)
        
        # Overfitting
        over_title = Text("Overfitting", font_size=28, color=PURPLE)
        over_subtitle = Text("High Variance", font_size=20, color=PURPLE)
        over_subtitle.next_to(over_title, DOWN)
        over_group = VGroup(over_title, over_subtitle)
        
        # Arrange horizontally
        scenarios = VGroup(under_group, optimal_group, over_group)
        scenarios.arrange(RIGHT, buff=1.5)
        scenarios.shift(DOWN * 0.5)
        
        self.play(LaggedStart(*[FadeIn(s) for s in scenarios], lag_ratio=0.3))
        self.wait()
        
        # Create mini plots for each scenario
        for i, (scenario, degree, color) in enumerate([
            (under_group, 1, RED),
            (optimal_group, 3, GREEN),
            (over_group, 15, PURPLE)
        ]):
            # Mini axes
            mini_axes = Axes(
                x_range=[-2, 2, 1],
                y_range=[-5, 5, 2],
                x_length=3,
                y_length=2,
                axis_config={"color": GRAY, "stroke_width": 2}
            )
            mini_axes.next_to(scenario, DOWN, buff=0.5)
            
            # Generate data
            x_data = np.linspace(-1.5, 1.5, 20)
            y_true = x_data**3
            y_noisy = y_true + np.random.normal(0, 0.5, len(x_data))
            
            # Data points
            dots = VGroup(*[
                Dot(mini_axes.c2p(x, y), color=YELLOW, radius=0.03)
                for x, y in zip(x_data, y_noisy)
            ])
            
            # Fit curve
            coeffs = np.polyfit(x_data, y_noisy, degree)
            poly_func = np.poly1d(coeffs)
            
            curve = mini_axes.plot(
                lambda x: poly_func(x),
                color=color,
                x_range=[-1.5, 1.5]
            )
            
            self.play(
                Create(mini_axes),
                LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.02),
                run_time=1
            )
            self.play(Create(curve), run_time=1)
        
        self.wait(2)
        
        # Show error bars
        error_text = Text(
            "Training Error vs Test Error",
            font_size=32,
            color=YELLOW
        )
        error_text.to_edge(DOWN)
        self.play(Write(error_text))
        self.wait(2)


class RegularizationScene(Scene):
    """Visualize regularization effect"""
    
    def construct(self):
        # Title
        title = Text("Regularization", font_size=48, weight=BOLD)
        subtitle = Text("Controlling Model Complexity", font_size=28)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Create comparison: No Regularization vs Regularization
        axes_left = Axes(
            x_range=[-3, 3, 1],
            y_range=[-8, 8, 4],
            x_length=5,
            y_length=4,
            axis_config={"color": BLUE}
        ).shift(LEFT * 3.5)
        
        axes_right = Axes(
            x_range=[-3, 3, 1],
            y_range=[-8, 8, 4],
            x_length=5,
            y_length=4,
            axis_config={"color": BLUE}
        ).shift(RIGHT * 3.5)
        
        # Labels
        left_label = Text("Without Regularization", font_size=24)
        left_label.next_to(axes_left, UP)
        
        right_label = Text("With Regularization", font_size=24)
        right_label.next_to(axes_right, UP)
        
        self.play(
            Create(axes_left), Create(axes_right),
            Write(left_label), Write(right_label)
        )
        
        # Generate data
        np.random.seed(42)
        x_data = np.linspace(-2.5, 2.5, 25)
        y_true = 0.5 * x_data**3 - x_data**2 + x_data
        y_noisy = y_true + np.random.normal(0, 1.2, len(x_data))
        
        # Data points on both axes
        dots_left = VGroup(*[
            Dot(axes_left.c2p(x, y), color=YELLOW, radius=0.05)
            for x, y in zip(x_data, y_noisy)
        ])
        
        dots_right = VGroup(*[
            Dot(axes_right.c2p(x, y), color=YELLOW, radius=0.05)
            for x, y in zip(x_data, y_noisy)
        ])
        
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in dots_left], lag_ratio=0.03),
            LaggedStart(*[GrowFromCenter(dot) for dot in dots_right], lag_ratio=0.03)
        )
        
        # Fit high-degree polynomial without regularization (overfitting)
        coeffs_overfit = np.polyfit(x_data, y_noisy, 12)
        poly_overfit = np.poly1d(coeffs_overfit)
        
        curve_left = axes_left.plot(
            lambda x: np.clip(poly_overfit(x), -8, 8),
            color=RED,
            x_range=[-2.5, 2.5]
        )
        
        # Simulate regularized fit (smoother)
        coeffs_reg = np.polyfit(x_data, y_noisy, 3)
        poly_reg = np.poly1d(coeffs_reg)
        
        curve_right = axes_right.plot(
            lambda x: poly_reg(x),
            color=GREEN,
            x_range=[-2.5, 2.5]
        )
        
        self.play(Create(curve_left), Create(curve_right))
        self.wait()
        
        # Highlight the difference
        overfit_warning = Text("Overfitted!", font_size=28, color=RED)
        overfit_warning.next_to(axes_left, DOWN)
        
        good_fit = Text("Good Fit", font_size=28, color=GREEN)
        good_fit.next_to(axes_right, DOWN)
        
        self.play(Write(overfit_warning), Write(good_fit))
        self.wait(2)
        
        # Show regularization formula
        self.play(FadeOut(VGroup(
            axes_left, axes_right, dots_left, dots_right,
            curve_left, curve_right, left_label, right_label,
            overfit_warning, good_fit
        )))
        
        # Ridge formula
        ridge_title = Text("Ridge Regression (L2)", font_size=40, color=BLUE)
        ridge_formula = MathTex(
            r"\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2",
            font_size=36
        )
        ridge_formula.next_to(ridge_title, DOWN, buff=0.5)
        
        self.play(Write(ridge_title))
        self.play(Write(ridge_formula))
        self.wait(2)
        self.play(FadeOut(ridge_title), FadeOut(ridge_formula))
        
        # Lasso formula
        lasso_title = Text("Lasso Regression (L1)", font_size=40, color=GREEN)
        lasso_formula = MathTex(
            r"\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|",
            font_size=36
        )
        lasso_formula.next_to(lasso_title, DOWN, buff=0.5)
        
        self.play(Write(lasso_title))
        self.play(Write(lasso_formula))
        self.wait(2)


class GradientDescentScene(Scene):
    """Visualize gradient descent for polynomial regression"""
    
    def construct(self):
        # Title
        title = Text("Gradient Descent", font_size=48, weight=BOLD)
        subtitle = Text("Finding Optimal Parameters", font_size=28)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Create 3D-like loss surface (simplified to 2D)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE}
        )
        
        x_label = axes.get_x_axis_label(r"\theta", direction=DOWN)
        y_label = axes.get_y_axis_label(r"Loss", edge=LEFT, direction=LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Loss function (parabola-like)
        loss_curve = axes.plot(
            lambda x: (x - 0.5)**2 + 1,
            color=YELLOW,
            x_range=[-2.5, 3]
        )
        
        self.play(Create(loss_curve))
        self.wait()
        
        # Starting point
        start_x = -2
        start_y = (start_x - 0.5)**2 + 1
        
        dot = Dot(axes.c2p(start_x, start_y), color=RED, radius=0.1)
        dot_label = Text("Start", font_size=24, color=RED)
        dot_label.next_to(dot, UP)
        
        self.play(GrowFromCenter(dot), Write(dot_label))
        self.wait()
        
        # Animate gradient descent steps
        learning_rate = 0.3
        current_x = start_x
        
        for step in range(15):
            # Calculate gradient (derivative)
            gradient = 2 * (current_x - 0.5)
            
            # Update parameter
            new_x = current_x - learning_rate * gradient
            new_y = (new_x - 0.5)**2 + 1
            
            # Create arrow showing direction
            arrow = Arrow(
                start=axes.c2p(current_x, (current_x - 0.5)**2 + 1),
                end=axes.c2p(new_x, new_y),
                color=GREEN,
                buff=0,
                stroke_width=4
            )
            
            self.play(
                GrowArrow(arrow),
                dot.animate.move_to(axes.c2p(new_x, new_y)),
                dot_label.animate.next_to(axes.c2p(new_x, new_y), UP),
                run_time=0.3
            )
            
            self.play(FadeOut(arrow), run_time=0.2)
            
            current_x = new_x
            
            # Stop if close to minimum
            if abs(gradient) < 0.1:
                break
        
        # Mark optimal point
        optimal_label = Text("Optimum", font_size=24, color=GREEN)
        optimal_label.next_to(dot, DOWN)
        self.play(
            dot.animate.set_color(GREEN),
            Transform(dot_label, optimal_label)
        )
        self.wait(2)
        
        # Show convergence message
        converge_text = Text(
            "Converged to minimum loss!",
            font_size=32,
            color=GREEN
        )
        converge_text.to_edge(DOWN)
        self.play(Write(converge_text))
        self.wait(2)


class FullRegressionDemo(Scene):
    """Complete demonstration combining all concepts"""
    
    def construct(self):
        # Main title
        title = Text(
            "Polynomial Regression & Regularization",
            font_size=52,
            weight=BOLD,
            gradient=(BLUE, GREEN)
        )
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_edge(UP))
        
        # Show key concepts
        concepts = VGroup(
            Text("1. Polynomial Features: x, x², x³, ...", font_size=28),
            Text("2. Bias-Variance Tradeoff", font_size=28),
            Text("3. Ridge (L2) Regularization", font_size=28),
            Text("4. Lasso (L1) Regularization", font_size=28),
            Text("5. Cross-Validation", font_size=28)
        )
        concepts.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concepts.shift(DOWN * 0.5)
        
        for concept in concepts:
            self.play(FadeIn(concept, shift=RIGHT), run_time=0.5)
        
        self.wait(2)
        
        # Fade out and show formula summary
        self.play(FadeOut(concepts))
        
        formulas = VGroup(
            MathTex(r"\text{Linear: } y = \beta_0 + \beta_1 x", font_size=32),
            MathTex(r"\text{Polynomial: } y = \beta_0 + \beta_1 x + \beta_2 x^2 + ...", font_size=32),
            MathTex(r"\text{Ridge: } + \lambda \sum \beta_j^2", font_size=32, color=BLUE),
            MathTex(r"\text{Lasso: } + \lambda \sum |\beta_j|", font_size=32, color=GREEN)
        )
        formulas.arrange(DOWN, buff=0.5)
        
        for formula in formulas:
            self.play(Write(formula), run_time=1)
        
        self.wait(2)


# Configuration for rendering
# To render: manim -pql this_file.py SceneName
# -p: preview, -ql: quality low (use -qh for high quality)

"""
RENDERING INSTRUCTIONS:
=======================

1. Install Manim:
   pip install manim

2. Run individual scenes:
   manim -pql manim_animations.py PolynomialFitScene
   manim -pql manim_animations.py BiasVarianceScene
   manim -pql manim_animations.py RegularizationScene
   manim -pql manim_animations.py GradientDescentScene
   manim -pql manim_animations.py FullRegressionDemo

3. Quality options:
   -ql : Low quality (480p, 15fps) - Fast preview
   -qm : Medium quality (720p, 30fps)
   -qh : High quality (1080p, 60fps) - For final videos
   -qk : 4K quality (2160p, 60fps)

4. Other options:
   -p : Preview after rendering
   -s : Save last frame as image
   -a : Render all scenes in file

ANIMATION DESCRIPTIONS:
=======================

PolynomialFitScene:
- Shows data points
- Fits polynomials of degrees 1, 2, 3, and 10
- Demonstrates how higher degrees fit data better

BiasVarianceScene:
- Compares underfitting, optimal, and overfitting
- Shows mini plots for each scenario
- Illustrates training vs test error

RegularizationScene:
- Compares models with and without regularization
- Shows Ridge (L2) and Lasso (L1) formulas
- Demonstrates smoother fits with regularization

GradientDescentScene:
- Visualizes parameter optimization
- Animates gradient descent steps
- Shows convergence to minimum loss

FullRegressionDemo:
- Summary of all key concepts
- Shows important formulas
- Comprehensive overview
"""