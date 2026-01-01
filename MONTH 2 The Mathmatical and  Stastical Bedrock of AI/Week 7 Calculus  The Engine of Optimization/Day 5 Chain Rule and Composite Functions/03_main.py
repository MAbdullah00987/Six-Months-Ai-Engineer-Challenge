
#ANIM ANIMATIONS FOR CHAIN RULE
#Create beautiful animations to visualize chain rule concepts
#Topic 4: Animated Visualizations with Manim
#Installation: pip install manim
#To render animations, run from command line:
#    manim -pql script_name.py SceneName
#    
#    -p = preview after rendering
#    -ql = quality low (use -qh for high quality)


from manim import *
import numpy as np


# SCENE 1: BASIC CHAIN RULE CONCEPT

class ChainRuleBasic(Scene):
    """
    Visualize: d/dx[f(g(x))] = f'(g(x)) * g'(x)
    Example: d/dx[(x^2 + 1)^3]
    """
    
    def construct(self):
        # Title
        title = Text("Chain Rule: Single Variable", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Show the composite function
        func_text = MathTex(
            r"f(g(x)) = (x^2 + 1)^3",
            font_size=40
        )
        func_text.shift(UP * 2)
        self.play(Write(func_text))
        self.wait()
        
        # Break down into components
        inner = MathTex(r"g(x) = x^2 + 1", font_size=36, color=BLUE)
        inner.shift(UP * 0.5 + LEFT * 3)
        
        outer = MathTex(r"f(u) = u^3", font_size=36, color=GREEN)
        outer.shift(UP * 0.5 + RIGHT * 3)
        
        self.play(
            Write(inner),
            Write(outer)
        )
        self.wait()
        
        # Show chain rule formula
        chain_rule = MathTex(
            r"\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}",
            font_size=40,
            color=YELLOW
        )
        chain_rule.shift(DOWN * 0.5)
        self.play(Write(chain_rule))
        self.wait()
        
        # Compute derivatives
        dg_dx = MathTex(r"\frac{dg}{dx} = 2x", font_size=32, color=BLUE)
        dg_dx.shift(DOWN * 1.8 + LEFT * 3)
        
        df_dg = MathTex(r"\frac{df}{dg} = 3g^2", font_size=32, color=GREEN)
        df_dg.shift(DOWN * 1.8 + RIGHT * 3)
        
        self.play(
            Write(dg_dx),
            Write(df_dg)
        )
        self.wait()
        
        # Final result
        result = MathTex(
            r"\frac{df}{dx} = 3(x^2 + 1)^2 \cdot 2x = 6x(x^2 + 1)^2",
            font_size=36,
            color=RED
        )
        result.shift(DOWN * 3)
        
        # Draw box around result
        result_box = SurroundingRectangle(result, color=RED, buff=0.2)
        
        self.play(Write(result))
        self.play(Create(result_box))
        self.wait(2)



# SCENE 2: COMPUTATIONAL GRAPH

class ComputationalGraph(Scene):
    """
    Visualize forward and backward pass through computational graph
    """
    
    def construct(self):
        title = Text("Computational Graph", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create nodes
        x_node = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        x_label = MathTex("x", font_size=32).move_to(x_node)
        x_group = VGroup(x_node, x_label).shift(LEFT * 5)
        
        square_node = Circle(radius=0.5, color=PURPLE, fill_opacity=0.5)
        square_label = MathTex("x^2", font_size=28).move_to(square_node)
        square_group = VGroup(square_node, square_label).shift(LEFT * 2.5)
        
        add_node = Circle(radius=0.5, color=PURPLE, fill_opacity=0.5)
        add_label = MathTex("+1", font_size=28).move_to(add_node)
        add_group = VGroup(add_node, add_label).shift(LEFT * 0)
        
        cube_node = Circle(radius=0.5, color=PURPLE, fill_opacity=0.5)
        cube_label = MathTex("^3", font_size=28).move_to(cube_node)
        cube_group = VGroup(cube_node, cube_label).shift(RIGHT * 2.5)
        
        y_node = Circle(radius=0.5, color=RED, fill_opacity=0.5)
        y_label = MathTex("y", font_size=32).move_to(y_node)
        y_group = VGroup(y_node, y_label).shift(RIGHT * 5)
        
        # Create arrows for forward pass
        arrow1 = Arrow(x_node.get_right(), square_node.get_left(), buff=0.1)
        arrow2 = Arrow(square_node.get_right(), add_node.get_left(), buff=0.1)
        arrow3 = Arrow(add_node.get_right(), cube_node.get_left(), buff=0.1)
        arrow4 = Arrow(cube_node.get_right(), y_node.get_left(), buff=0.1)
        
        # Animate forward pass
        subtitle = Text("Forward Pass", font_size=36, color=GREEN)
        subtitle.shift(DOWN * 3)
        self.play(Write(subtitle))
        
        self.play(Create(x_group))
        self.play(GrowArrow(arrow1), Create(square_group))
        self.wait(0.5)
        self.play(GrowArrow(arrow2), Create(add_group))
        self.wait(0.5)
        self.play(GrowArrow(arrow3), Create(cube_group))
        self.wait(0.5)
        self.play(GrowArrow(arrow4), Create(y_group))
        self.wait()
        
        # Create backward arrows (gradients)
        back_arrow1 = Arrow(square_node.get_left(), x_node.get_right(), 
                           buff=0.1, color=YELLOW).shift(DOWN * 0.3)
        back_arrow2 = Arrow(add_node.get_left(), square_node.get_right(), 
                           buff=0.1, color=YELLOW).shift(DOWN * 0.3)
        back_arrow3 = Arrow(cube_node.get_left(), add_node.get_right(), 
                           buff=0.1, color=YELLOW).shift(DOWN * 0.3)
        back_arrow4 = Arrow(y_node.get_left(), cube_node.get_right(), 
                           buff=0.1, color=YELLOW).shift(DOWN * 0.3)
        
        # Gradient labels
        grad4 = MathTex(r"\frac{\partial y}{\partial y}=1", font_size=20, color=YELLOW)
        grad4.next_to(back_arrow4, DOWN, buff=0.1)
        
        grad3 = MathTex(r"3u^2", font_size=20, color=YELLOW)
        grad3.next_to(back_arrow3, DOWN, buff=0.1)
        
        grad2 = MathTex(r"1", font_size=20, color=YELLOW)
        grad2.next_to(back_arrow2, DOWN, buff=0.1)
        
        grad1 = MathTex(r"2x", font_size=20, color=YELLOW)
        grad1.next_to(back_arrow1, DOWN, buff=0.1)
        
        # Animate backward pass
        self.play(FadeOut(subtitle))
        subtitle2 = Text("Backward Pass (Gradients)", font_size=36, color=YELLOW)
        subtitle2.shift(DOWN * 3)
        self.play(Write(subtitle2))
        
        self.play(GrowArrow(back_arrow4), Write(grad4))
        self.wait(0.5)
        self.play(GrowArrow(back_arrow3), Write(grad3))
        self.wait(0.5)
        self.play(GrowArrow(back_arrow2), Write(grad2))
        self.wait(0.5)
        self.play(GrowArrow(back_arrow1), Write(grad1))
        self.wait(2)



# SCENE 3: GRADIENT DESCENT VISUALIZATION


class GradientDescentVisualization(Scene):
    """
    Animate gradient descent using chain rule
    """
    
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[0, 20, 5],
            x_length=8,
            y_length=5,
            axis_config={"color": WHITE},
        )
        
        # Function: f(x) = (x^2 + 1)^2
        graph = axes.plot(
            lambda x: (x**2 + 1)**2,
            color=BLUE,
            x_range=[-2, 2]
        )
        
        # Labels
        title = Text("Gradient Descent with Chain Rule", font_size=40)
        title.to_edge(UP)
        
        func_label = MathTex(r"f(x) = (x^2 + 1)^2", font_size=32)
        func_label.next_to(axes, UP)
        
        self.play(Write(title))
        self.play(Create(axes), Write(func_label))
        self.play(Create(graph))
        self.wait()
        
        # Starting point
        x_val = 1.5
        learning_rate = 0.1
        
        # Create moving dot
        dot = Dot(color=RED, radius=0.1)
        dot.move_to(axes.c2p(x_val, (x_val**2 + 1)**2))
        
        # Value tracker
        x_tracker = ValueTracker(x_val)
        
        # Update dot position
        dot.add_updater(
            lambda m: m.move_to(
                axes.c2p(
                    x_tracker.get_value(),
                    (x_tracker.get_value()**2 + 1)**2
                )
            )
        )
        
        self.play(FadeIn(dot))
        
        # Gradient descent steps
        for step in range(15):
            x = x_tracker.get_value()
            # Derivative: f'(x) = 2(x^2 + 1) * 2x = 4x(x^2 + 1)
            grad = 4 * x * (x**2 + 1)
            x_new = x - learning_rate * grad
            
            # Show gradient arrow
            arrow = Arrow(
                start=axes.c2p(x, (x**2 + 1)**2),
                end=axes.c2p(x - 0.3 * grad / abs(grad), (x**2 + 1)**2),
                color=YELLOW,
                buff=0
            )
            
            grad_text = MathTex(
                f"\\nabla f = {grad:.2f}",
                font_size=24,
                color=YELLOW
            )
            grad_text.next_to(arrow, DOWN)
            
            self.play(GrowArrow(arrow), Write(grad_text), run_time=0.3)
            self.play(
                x_tracker.animate.set_value(x_new),
                FadeOut(arrow),
                FadeOut(grad_text),
                run_time=0.5
            )
            
            if abs(grad) < 0.1:
                break
        
        # Final text
        final_text = Text("Converged to minimum!", font_size=32, color=GREEN)
        final_text.shift(DOWN * 3)
        self.play(Write(final_text))
        self.wait(2)



# SCENE 4: NEURAL NETWORK BACKPROPAGATION


class NeuralNetworkBackprop(Scene):
    """
    Visualize backpropagation in a simple neural network
    """
    
    def construct(self):
        title = Text("Neural Network Backpropagation", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create layers
        # Input layer (2 neurons)
        input_neurons = VGroup(*[
            Circle(radius=0.3, color=BLUE, fill_opacity=0.7)
            for _ in range(2)
        ]).arrange(DOWN, buff=0.8).shift(LEFT * 5)
        
        input_labels = VGroup(*[
            MathTex(f"x_{i+1}", font_size=24).move_to(neuron)
            for i, neuron in enumerate(input_neurons)
        ])
        
        # Hidden layer (3 neurons)
        hidden_neurons = VGroup(*[
            Circle(radius=0.3, color=PURPLE, fill_opacity=0.7)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.6).shift(LEFT * 1)
        
        hidden_labels = VGroup(*[
            MathTex(f"h_{i+1}", font_size=24).move_to(neuron)
            for i, neuron in enumerate(hidden_neurons)
        ])
        
        # Output layer (1 neuron)
        output_neuron = Circle(radius=0.3, color=RED, fill_opacity=0.7)
        output_neuron.shift(RIGHT * 4)
        output_label = MathTex("y", font_size=24).move_to(output_neuron)
        
        # Draw network
        self.play(
            *[Create(n) for n in input_neurons],
            *[Write(l) for l in input_labels]
        )
        self.wait(0.5)
        
        # Create connections (weights)
        connections1 = VGroup()
        for inp in input_neurons:
            for hid in hidden_neurons:
                line = Line(inp.get_right(), hid.get_left(), stroke_width=1)
                connections1.add(line)
        
        connections2 = VGroup()
        for hid in hidden_neurons:
            line = Line(hid.get_right(), output_neuron.get_left(), stroke_width=1)
            connections2.add(line)
        
        self.play(*[Create(c) for c in connections1])
        self.play(
            *[Create(n) for n in hidden_neurons],
            *[Write(l) for l in hidden_labels]
        )
        self.wait(0.5)
        
        self.play(*[Create(c) for c in connections2])
        self.play(Create(output_neuron), Write(output_label))
        self.wait()
        
        # Forward pass animation
        subtitle = Text("Forward Pass", font_size=30, color=GREEN)
        subtitle.shift(DOWN * 3)
        self.play(Write(subtitle))
        
        # Animate signal propagation
        for inp in input_neurons:
            for hid in hidden_neurons:
                line = Line(inp.get_right(), hid.get_left(), color=GREEN, stroke_width=3)
                self.play(Create(line), run_time=0.2)
                self.play(FadeOut(line), run_time=0.1)
        
        for hid in hidden_neurons:
            line = Line(hid.get_right(), output_neuron.get_left(), color=GREEN, stroke_width=3)
            self.play(Create(line), run_time=0.2)
            self.play(FadeOut(line), run_time=0.1)
        
        self.wait()
        
        # Backward pass animation
        self.play(FadeOut(subtitle))
        subtitle2 = Text("Backward Pass (Gradients)", font_size=30, color=YELLOW)
        subtitle2.shift(DOWN * 3)
        self.play(Write(subtitle2))
        
        # Animate gradient backpropagation
        for hid in hidden_neurons:
            line = Line(output_neuron.get_left(), hid.get_right(), color=YELLOW, stroke_width=3)
            self.play(Create(line), run_time=0.2)
            self.play(FadeOut(line), run_time=0.1)
        
        for hid in hidden_neurons:
            for inp in input_neurons:
                line = Line(hid.get_left(), inp.get_right(), color=YELLOW, stroke_width=3)
                self.play(Create(line), run_time=0.2)
                self.play(FadeOut(line), run_time=0.1)
        
        self.wait(2)


# INSTRUCTIONS FOR RUNNING

"""
TO RENDER THESE ANIMATIONS:
1. Install Manim:
   pip install manim
2. Save this file as 'chain_rule_manim.py'
3. Render individual scenes:
   manim -pql chain_rule_manim.py ChainRuleBasic
   manim -pql chain_rule_manim.py ComputationalGraph
   manim -pql chain_rule_manim.py GradientDescentVisualization
   manim -pql chain_rule_manim.py NeuralNetworkBackprop

4. For high quality:
   manim -pqh chain_rule_manim.py SceneName

5. To render all scenes:
   manim -pql chain_rule_manim.py -a

Flags:
-p : Preview after rendering
-q : Quality (l=low, m=medium, h=high, k=4K)
-a : Render all scenes
"""

print("\nAvailable scenes:")
print("1. ChainRuleBasic - Basic chain rule concept")
print("2. ComputationalGraph - Forward/backward pass visualization")
print("3. GradientDescentVisualization - Gradient descent in action")
print("4. NeuralNetworkBackprop - Full neural network animation")
print("\nRun: manim -pql this_file.py SceneName")
