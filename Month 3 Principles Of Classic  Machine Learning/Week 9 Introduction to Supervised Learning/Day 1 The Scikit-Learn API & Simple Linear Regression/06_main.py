
#Part 6: Manim - Animated ML Concepts



from manim import *
import numpy as np

# SCENE 1: Understanding Linear Regression

class LinearRegressionScene(Scene):
    def construct(self):
        # Title
        title = Text("Linear Regression: Finding the Best Fit Line", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 30, 5],
            x_length=8,
            y_length=5,
            axis_config={"color": BLUE, "include_numbers": True},
            tips=False
        ).shift(DOWN * 0.5)
        
        x_label = axes.get_x_axis_label("x", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generate data points
        np.random.seed(42)
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_data = 2 + 3 * x_data + np.random.randn(len(x_data)) * 2
        
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=YELLOW, radius=0.08)
            for x, y in zip(x_data, y_data)
        ])
        
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.1))
        self.wait()
        
        # Show several candidate lines
        equation_text = MathTex(r"\hat{y} = w_0 + w_1 x", font_size=32)
        equation_text.to_corner(UL).shift(DOWN * 0.5)
        self.play(Write(equation_text))
        
        # Bad fit
        bad_line = axes.plot(lambda x: 5 + 1.5 * x, color=RED, x_range=[0, 10])
        bad_label = Text("Bad fit", font_size=24, color=RED).next_to(bad_line, RIGHT)
        self.play(Create(bad_line), Write(bad_label))
        self.wait()
        
        self.play(FadeOut(bad_line), FadeOut(bad_label))
        
        # Good fit (true line)
        w0, w1 = np.polyfit(x_data, y_data, 1)
        best_line = axes.plot(lambda x: w0 + w1 * x, color=GREEN, x_range=[0, 10])
        best_label = Text("Best fit!", font_size=24, color=GREEN).next_to(best_line, RIGHT)
        self.play(Create(best_line), Write(best_label))
        
        # Show residuals
        residual_lines = VGroup(*[
            DashedLine(
                axes.c2p(x, y),
                axes.c2p(x, w0 + w1 * x),
                color=ORANGE,
                stroke_width=2
            )
            for x, y in zip(x_data, y_data)
        ])
        
        residual_text = Text("Residuals (errors)", font_size=28, color=ORANGE)
        residual_text.to_corner(UR).shift(DOWN * 0.5)
        
        self.play(Create(residual_lines), Write(residual_text))
        self.wait(2)
        
        # Show cost function
        cost_formula = MathTex(
            r"J(w_0, w_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            font_size=32
        )
        cost_formula.to_edge(DOWN)
        
        self.play(Transform(residual_text, cost_formula))
        self.wait(3)

# ============================================================================
# SCENE 2: Gradient Descent Visualization
# ============================================================================
class GradientDescentScene(Scene):
    def construct(self):
        title = Text("Gradient Descent: Optimizing the Loss", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create 2D loss surface
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 30, 5],
            x_length=10,
            y_length=5,
            axis_config={"color": BLUE},
            tips=False
        ).shift(DOWN * 0.5)
        
        x_label = axes.get_x_axis_label(r"w", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label(r"Loss", edge=UP, direction=UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Loss function (parabola)
        loss_curve = axes.plot(lambda x: x**2 + 1, color=WHITE, x_range=[-4, 4])
        self.play(Create(loss_curve))
        
        # Starting point
        start_w = 3.5
        start_loss = start_w**2 + 1
        point = Dot(axes.c2p(start_w, start_loss), color=YELLOW, radius=0.12)
        self.play(GrowFromCenter(point))
        
        # Gradient descent steps
        w = start_w
        alpha = 0.3  # learning rate
        
        for step in range(8):
            # Compute gradient
            grad = 2 * w
            w_new = w - alpha * grad
            loss_new = w_new**2 + 1
            
            # Draw tangent line
            tangent_slope = grad
            tangent = axes.plot(
                lambda x: tangent_slope * (x - w) + (w**2 + 1),
                color=RED,
                x_range=[w - 1, w + 1]
            )
            
            # Arrow showing direction
            arrow = Arrow(
                axes.c2p(w, w**2 + 1),
                axes.c2p(w_new, w**2 + 1),
                color=GREEN,
                buff=0,
                stroke_width=4
            )
            
            self.play(Create(tangent), Create(arrow), run_time=0.5)
            self.wait(0.3)
            
            # Move point
            new_point = Dot(axes.c2p(w_new, loss_new), color=YELLOW, radius=0.12)
            self.play(
                Transform(point, new_point),
                FadeOut(tangent),
                FadeOut(arrow),
                run_time=0.5
            )
            
            w = w_new
            
            if abs(w) < 0.1:
                break
        
        # Final message
        converged = Text("Converged to minimum!", font_size=32, color=GREEN)
        converged.to_edge(DOWN)
        self.play(Write(converged))
        self.wait(2)

# ============================================================================
# SCENE 3: Feature Scaling Effect
# ============================================================================
class FeatureScalingScene(Scene):
    def construct(self):
        title = Text("Feature Scaling: Why It Matters", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Split screen
        left_title = Text("Without Scaling", font_size=28, color=RED)
        left_title.move_to(LEFT * 3 + UP * 2)
        
        right_title = Text("With Scaling", font_size=28, color=GREEN)
        right_title.move_to(RIGHT * 3 + UP * 2)
        
        self.play(Write(left_title), Write(right_title))
        
        # Left: Elongated contours (unscaled features)
        left_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3,
            y_length=3,
            axis_config={"color": BLUE, "stroke_width": 2},
            tips=False
        ).move_to(LEFT * 3 + DOWN * 0.5)
        
        # Right: Circular contours (scaled features)
        right_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3,
            y_length=3,
            axis_config={"color": BLUE, "stroke_width": 2},
            tips=False
        ).move_to(RIGHT * 3 + DOWN * 0.5)
        
        self.play(Create(left_axes), Create(right_axes))
        
        # Contour ellipses (left)
        ellipses_left = VGroup(*[
            Ellipse(width=0.5 + i*0.5, height=1.5 + i*0.5, color=YELLOW)
            .move_to(left_axes.c2p(0, 0))
            for i in range(4)
        ])
        
        # Contour circles (right)
        circles_right = VGroup(*[
            Circle(radius=0.3 + i*0.3, color=YELLOW)
            .move_to(right_axes.c2p(0, 0))
            for i in range(4)
        ])
        
        self.play(Create(ellipses_left), Create(circles_right))
        
        # Gradient descent path (zigzag left, direct right)
        left_path_points = [
            [1.5, 1.5], [1.2, -1.0], [0.8, 0.8],
            [0.5, -0.5], [0.2, 0.3], [0, 0]
        ]
        right_path_points = [
            [1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0, 0]
        ]
        
        left_path = VMobject(color=RED, stroke_width=4)
        left_path.set_points_smoothly([
            left_axes.c2p(x, y) for x, y in left_path_points
        ])
        
        right_path = VMobject(color=GREEN, stroke_width=4)
        right_path.set_points_smoothly([
            right_axes.c2p(x, y) for x, y in right_path_points
        ])
        
        left_dot = Dot(left_axes.c2p(1.5, 1.5), color=RED, radius=0.08)
        right_dot = Dot(right_axes.c2p(1.5, 1.5), color=GREEN, radius=0.08)
        
        self.play(GrowFromCenter(left_dot), GrowFromCenter(right_dot))
        
        # Animate both paths simultaneously
        self.play(
            MoveAlongPath(left_dot, left_path),
            MoveAlongPath(right_dot, right_path),
            Create(left_path),
            Create(right_path),
            run_time=4,
            rate_func=linear
        )
        
        # Conclusion
        left_steps = Text("Many steps", font_size=20, color=RED)
        left_steps.next_to(left_axes, DOWN)
        
        right_steps = Text("Few steps", font_size=20, color=GREEN)
        right_steps.next_to(right_axes, DOWN)
        
        self.play(Write(left_steps), Write(right_steps))
        self.wait(2)

# ============================================================================
# SCENE 4: Regularization Visualization
# ============================================================================
class RegularizationScene(Scene):
    def construct(self):
        title = Text("Regularization: Controlling Model Complexity", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create three subplots
        labels = ["No Regularization", "Ridge (L2)", "Lasso (L1)"]
        colors = [RED, BLUE, GREEN]
        
        axes_list = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            x_pos = -4 + i * 4
            
            ax = Axes(
                x_range=[0, 10, 2],
                y_range=[0, 10, 2],
                x_length=3,
                y_length=2.5,
                axis_config={"color": WHITE, "stroke_width": 1},
                tips=False
            ).move_to([x_pos, 0, 0])
            
            label_text = Text(label, font_size=20, color=color)
            label_text.next_to(ax, UP, buff=0.2)
            
            axes_list.append((ax, label_text, color))
        
        for ax, label, _ in axes_list:
            self.play(Create(ax), Write(label), run_time=0.7)
        
        # Generate data and models
        np.random.seed(42)
        x_data = np.linspace(1, 9, 20)
        y_data = 5 + 0.3 * x_data + np.random.randn(20) * 1.5
        
        # Plot data points on all axes
        dots_groups = []
        for ax, _, color in axes_list:
            dots = VGroup(*[
                Dot(ax.c2p(x, y), radius=0.04, color=YELLOW)
                for x, y in zip(x_data, y_data)
            ])
            dots_groups.append(dots)
            self.play(Create(dots), run_time=0.5)
        
        # Fit different models
        # 1. Overfit (polynomial)
        coeffs_overfit = np.polyfit(x_data, y_data, deg=10)
        poly_overfit = np.poly1d(coeffs_overfit)
        
        # 2. Ridge (moderate complexity)
        coeffs_ridge = np.polyfit(x_data, y_data, deg=3)
        poly_ridge = np.poly1d(coeffs_ridge)
        
        # 3. Lasso (simpler, like linear)
        coeffs_lasso = np.polyfit(x_data, y_data, deg=1)
        poly_lasso = np.poly1d(coeffs_lasso)
        
        models = [poly_overfit, poly_ridge, poly_lasso]
        
        # Plot model curves
        for (ax, _, color), model in zip(axes_list, models):
            curve = ax.plot(
                lambda x: np.clip(model(x), 0, 10),
                color=color,
                x_range=[1, 9]
            )
            self.play(Create(curve), run_time=1)
        
        # Add annotations
        annotations = [
            "Overfits data",
            "Balanced",
            "Underfits slightly"
        ]
        
        annotation_texts = []
        for (ax, _, _), annot in zip(axes_list, annotations):
            text = Text(annot, font_size=16, color=WHITE)
            text.next_to(ax, DOWN, buff=0.2)
            annotation_texts.append(text)
            self.play(Write(text), run_time=0.5)
        
        self.wait(3)

# ============================================================================
# SCENE 5: Train/Test Split Animation
# ============================================================================
class TrainTestSplitScene(Scene):
    def construct(self):
        title = Text("Train/Test Split: Avoiding Overfitting", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create dataset visualization
        n_samples = 20
        dots = VGroup(*[
            Dot([i * 0.5 - 4.5, 0, 0], radius=0.15, color=WHITE)
            for i in range(n_samples)
        ])
        
        dataset_label = Text("Complete Dataset", font_size=24)
        dataset_label.next_to(dots, UP, buff=0.5)
        
        self.play(Write(dataset_label))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.05))
        self.wait()
        
        # Split into train and test
        split_idx = int(0.8 * n_samples)
        
        train_dots = dots[:split_idx]
        test_dots = dots[split_idx:]
        
        # Move train set up
        self.play(
            train_dots.animate.shift(UP * 1.5).set_color(BLUE),
            run_time=1
        )
        train_label = Text("Training Set (80%)", font_size=20, color=BLUE)
        train_label.next_to(train_dots, UP, buff=0.3)
        self.play(Write(train_label))
        
        # Move test set down
        self.play(
            test_dots.animate.shift(DOWN * 1.5).set_color(RED),
            run_time=1
        )
        test_label = Text("Test Set (20%)", font_size=20, color=RED)
        test_label.next_to(test_dots, DOWN, buff=0.3)
        self.play(Write(test_label))
        
        self.wait()
        
        # Show training process
        train_box = SurroundingRectangle(train_dots, color=BLUE, buff=0.2)
        train_text = Text("Model learns from this", font_size=18, color=BLUE)
        train_text.next_to(train_box, LEFT, buff=0.5)
        
        self.play(Create(train_box), Write(train_text))
        self.wait()
        
        # Show testing process
        test_box = SurroundingRectangle(test_dots, color=RED, buff=0.2)
        test_text = Text("Model evaluated on this", font_size=18, color=RED)
        test_text.next_to(test_box, LEFT, buff=0.5)
        
        self.play(Create(test_box), Write(test_text))
        self.wait()
        
        # Key insight
        insight = Text(
            "Test set NEVER seen during training!",
            font_size=24,
            color=YELLOW
        )
        insight.to_edge(DOWN)
        self.play(Write(insight))
        self.wait(3)


# RENDERING INSTRUCTIONS


"""
DAY 1 - PART 6: Manim Animations for ML Concepts
Visual explanations of gradient descent, loss surfaces, and more

SETUP INSTRUCTIONS:
1. Install manim: pip install manim
2. Save this file as ml_animations.py
3. Render animations:
   - Single scene: manim ml_animations.py GradientDescentScene -pql
   - All scenes: manim ml_animations.py -pql

FLAGS:
-p: Preview after rendering
-q: Quality (l=low, m=medium, h=high)
"""

"""
To render all scenes:

1. Save this file as ml_animations.py

2. Run each scene:
   manim -pql ml_animations.py LinearRegressionScene
   manim -pql ml_animations.py GradientDescentScene
   manim -pql ml_animations.py FeatureScalingScene
   manim -pql ml_animations.py RegularizationScene
   manim -pql ml_animations.py TrainTestSplitScene

3. Or render all at once:
   manim -pql ml_animations.py

Flags:
-p: Preview when done
-q: Quality (l=480p, m=720p, h=1080p, k=4K)
-l: Low quality (fast render)

Output location: media/videos/ml_animations/480p15/
"""