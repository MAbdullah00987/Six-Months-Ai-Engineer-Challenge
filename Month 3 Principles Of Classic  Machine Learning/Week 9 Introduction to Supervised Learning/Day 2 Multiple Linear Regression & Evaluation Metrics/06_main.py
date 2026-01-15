
#Part 6. Advanced Visualization with Manim (Animated Mathematics)

"""
Manim Animations for Linear Regression
Run with: manim -pql script.py SceneName

Installation: pip install manim
"""

from manim import *
import numpy as np

class GradientDescentVisualization(Scene):
    """
    Visualize gradient descent optimization for simple linear regression
    """
    def construct(self):
        # Title
        title = Text("Gradient Descent for Linear Regression", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Generate data points
        np.random.seed(42)
        X_data = np.linspace(-2, 2, 20)
        y_data = 2 + 1.5 * X_data + np.random.randn(20) * 0.5
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 8, 2],
            x_length=6,
            y_length=5,
            axis_config={"include_tip": True}
        )
        axes.add_coordinates()
        
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Plot data points
        dots = VGroup(*[
            Dot(axes.coords_to_point(x, y), color=BLUE)
            for x, y in zip(X_data, y_data)
        ])
        
        self.play(Create(axes), Write(labels))
        self.play(Create(dots))
        self.wait()
        
        # Initial random line
        theta0, theta1 = 0, 0
        
        def get_line(t0, t1):
            return axes.plot(lambda x: t0 + t1 * x, color=RED, x_range=[-3, 3])
        
        regression_line = get_line(theta0, theta1)
        
        # Equation text
        equation = MathTex(f"y = {theta0:.2f} + {theta1:.2f}x")
        equation.next_to(axes, DOWN)
        
        self.play(Create(regression_line), Write(equation))
        self.wait()
        
        # Gradient descent iterations
        learning_rate = 0.1
        iterations = 20
        
        for i in range(iterations):
            # Compute gradient
            predictions = theta0 + theta1 * X_data
            errors = predictions - y_data
            
            grad_theta0 = np.mean(errors)
            grad_theta1 = np.mean(errors * X_data)
            
            # Update parameters
            theta0 -= learning_rate * grad_theta0
            theta1 -= learning_rate * grad_theta1
            
            # Create new line
            new_line = get_line(theta0, theta1)
            new_equation = MathTex(f"y = {theta0:.2f} + {theta1:.2f}x")
            new_equation.next_to(axes, DOWN)
            
            # Animate transformation
            self.play(
                Transform(regression_line, new_line),
                Transform(equation, new_equation),
                run_time=0.3
            )
        
        self.wait(2)


class CostFunctionSurface(ThreeDScene):
    """
    3D visualization of the cost function J(θ₀, θ₁)
    """
    def construct(self):
        # Title
        title = Text("Cost Function J(θ)", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.linspace(-2, 2, 30)
        y = 2 + 1.5 * X + np.random.randn(30) * 0.5
        
        # Define cost function
        def cost_function(theta0, theta1):
            predictions = theta0 + theta1 * X
            cost = np.mean((predictions - y)**2)
            return cost
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-2, 6, 1],
            y_range=[-1, 4, 1],
            z_range=[0, 20, 5],
            x_length=8,
            y_length=8,
            z_length=6
        )
        
        labels = axes.get_axis_labels(
            x_label=MathTex("\\theta_0"),
            y_label=MathTex("\\theta_1"),
            z_label=MathTex("J(\\theta)")
        )
        
        # Create surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, cost_function(u, v)),
            u_range=[-2, 6],
            v_range=[-1, 4],
            resolution=(30, 30),
            fill_opacity=0.7
        )
        surface.set_color_by_gradient(BLUE, GREEN, YELLOW, RED)
        
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        
        self.play(Create(axes), Write(labels))
        self.play(Create(surface))
        
        # Animate camera rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        self.wait(2)


class ResidualVisualization(Scene):
    """
    Visualize residuals (errors) between predictions and actual values
    """
    def construct(self):
        title = Text("Residuals: ε = y - ŷ", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Data
        np.random.seed(42)
        X = np.linspace(-2, 2, 10)
        y = 2 + 1.5 * X + np.random.randn(10) * 0.5
        
        # Fitted line parameters
        theta0, theta1 = 2, 1.5
        y_pred = theta0 + theta1 * X
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 8, 2],
            x_length=8,
            y_length=6
        )
        axes.add_coordinates()
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Regression line
        line = axes.plot(lambda x: theta0 + theta1 * x, color=RED)
        
        # Data points
        dots = VGroup(*[
            Dot(axes.coords_to_point(x_val, y_val), color=BLUE)
            for x_val, y_val in zip(X, y)
        ])
        
        self.play(Create(axes), Write(labels))
        self.play(Create(line))
        self.play(Create(dots))
        self.wait()
        
        # Draw residual lines
        residual_lines = VGroup()
        residual_texts = VGroup()
        
        for i, (x_val, y_val) in enumerate(zip(X, y)):
            pred_point = axes.coords_to_point(x_val, y_pred[i])
            actual_point = axes.coords_to_point(x_val, y_val)
            
            residual = y_val - y_pred[i]
            
            # Vertical line showing residual
            res_line = DashedLine(
                pred_point, actual_point,
                color=YELLOW if residual > 0 else GREEN
            )
            residual_lines.add(res_line)
            
            # Text showing residual value
            res_text = MathTex(f"{residual:.2f}", font_size=20)
            res_text.next_to(res_line, RIGHT, buff=0.1)
            residual_texts.add(res_text)
        
        self.play(Create(residual_lines), run_time=2)
        self.play(Write(residual_texts), run_time=2)
        self.wait(2)
        
        # Show MSE formula
        mse_formula = MathTex(
            "MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2",
            font_size=40
        )
        mse_formula.to_edge(DOWN)
        
        self.play(Write(mse_formula))
        self.wait(2)


class NormalEquationVisualization(Scene):
    """
    Visualize the Normal Equation solution
    """
    def construct(self):
        title = Text("Normal Equation: Closed-Form Solution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Show derivation step by step
        equations = VGroup(
            MathTex("J(\\theta) = \\frac{1}{2m}(X\\theta - y)^T(X\\theta - y)"),
            MathTex("\\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m}X^T(X\\theta - y)"),
            MathTex("X^T(X\\theta - y) = 0"),
            MathTex("X^TX\\theta = X^Ty"),
            MathTex("\\theta = (X^TX)^{-1}X^Ty", color=YELLOW)
        ).arrange(DOWN, buff=0.7)
        
        for eq in equations:
            self.play(Write(eq))
            self.wait(0.5)
        
        self.wait(2)
        
        # Highlight final equation
        box = SurroundingRectangle(equations[-1], color=YELLOW, buff=0.2)
        self.play(Create(box))
        self.wait(2)


class RegressionMetrics(Scene):
    """
    Visualize different evaluation metrics
    """
    def construct(self):
        title = Text("Evaluation Metrics", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create metric boxes
        metrics = VGroup(
            VGroup(
                Text("MAE", font_size=30, color=BLUE),
                MathTex("\\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|")
            ).arrange(DOWN),
            VGroup(
                Text("MSE", font_size=30, color=GREEN),
                MathTex("\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2")
            ).arrange(DOWN),
            VGroup(
                Text("RMSE", font_size=30, color=YELLOW),
                MathTex("\\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}")
            ).arrange(DOWN),
            VGroup(
                Text("R²", font_size=30, color=RED),
                MathTex("1 - \\frac{SS_{res}}{SS_{tot}}")
            ).arrange(DOWN)
        ).arrange_in_grid(rows=2, cols=2, buff=1.5)
        
        metrics.move_to(ORIGIN)
        
        for metric in metrics:
            box = SurroundingRectangle(metric, buff=0.3)
            self.play(Create(box), Write(metric))
            self.wait(0.5)
        
        self.wait(2)


# Instructions to render:
"""
To render these animations, save this file as 'linear_regression_manim.py' and run:

# Render all scenes in low quality (fast preview)
manim -pql linear_regression_manim.py

# Render specific scene in high quality
manim -pqh linear_regression_manim.py GradientDescentVisualization

# Available scenes:
1. GradientDescentVisualization - Shows gradient descent fitting a line
2. CostFunctionSurface - 3D visualization of cost function
3. ResidualVisualization - Shows residuals between predictions and actual
4. NormalEquationVisualization - Derives normal equation
5. RegressionMetrics - Shows all evaluation metrics

Quality flags:
-ql : Low quality (854x480, 15fps) - fast preview
-qm : Medium quality (1280x720, 30fps)
-qh : High quality (1920x1080, 60fps)
-qk : 4K quality (3840x2160, 60fps)

Flags:
-p : Preview (automatically opens video after rendering)
-s : Save last frame as image
-a : Render all scenes in the file
"""