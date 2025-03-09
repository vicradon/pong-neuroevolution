import fs from "fs";
import path from "path";
import { Neuroevolution } from "./Neuroevolution.js";

// Initialize Neuroevolution with configuration
const Neuro = new Neuroevolution({
  population: 20, // 20 networks per generation
  network: [6, [8], 1], // 6 inputs, 1 hidden layer with 8 neurons, 1 output
  mutationRate: 0.2, // Chance of mutation
  elitism: 0.2, // Keep top 20% for next generation
  activation: function (a) {
    return 1 / (1 + Math.exp(-a)); // Sigmoid function
  },
});

// Simple simulated Pong environment
class PongSimulation {
  constructor() {
    // Canvas dimensions
    this.width = 1200;
    this.height = 700;

    // Paddle properties
    this.paddleHeight = 100;
    this.paddleWidth = 20;
    this.paddleSpeed = 5;

    // Ball properties
    this.ballRadius = 10;
    this.ballSpeedX = 8;
    this.ballSpeedY = 2;

    this.reset();
  }

  reset() {
    // Reset ball to center with random direction
    this.ball = {
      x: this.width / 2,
      y: this.height / 2,
      dx: (Math.random() > 0.5 ? 1 : -1) * this.ballSpeedX,
      dy: (Math.random() > 0.5 ? 1 : -1) * Math.random() * this.ballSpeedY,
    };

    // Reset paddles
    this.leftPaddle = {
      y: this.height / 2 - this.paddleHeight / 2,
      x: 20,
    };

    this.rightPaddle = {
      y: this.height / 2 - this.paddleHeight / 2,
      x: this.width - 40,
    };

    this.gameOver = false;
  }

  // Check for collision between ball and paddle
  checkCollision(paddle) {
    return (
      this.ball.x - this.ballRadius <= paddle.x + this.paddleWidth &&
      this.ball.x + this.ballRadius >= paddle.x &&
      this.ball.y >= paddle.y &&
      this.ball.y <= paddle.y + this.paddleHeight
    );
  }

  // Update game state for one frame
  update(leftPaddleMove, rightPaddleMove) {
    // Move ball
    this.ball.x += this.ball.dx;
    this.ball.y += this.ball.dy;

    // Move paddles based on AI decisions
    // leftPaddleMove and rightPaddleMove should be values between 0 and 1
    if (leftPaddleMove < 0.5 && this.leftPaddle.y > 0) {
      this.leftPaddle.y -= this.paddleSpeed;
    } else if (
      leftPaddleMove >= 0.5 &&
      this.leftPaddle.y < this.height - this.paddleHeight
    ) {
      this.leftPaddle.y += this.paddleSpeed;
    }

    if (rightPaddleMove < 0.5 && this.rightPaddle.y > 0) {
      this.rightPaddle.y -= this.paddleSpeed;
    } else if (
      rightPaddleMove >= 0.5 &&
      this.rightPaddle.y < this.height - this.paddleHeight
    ) {
      this.rightPaddle.y += this.paddleSpeed;
    }

    // Handle collisions with paddles
    if (
      this.checkCollision(this.leftPaddle) ||
      this.checkCollision(this.rightPaddle)
    ) {
      this.ball.dx *= -1;
    }

    // Handle collisions with top and bottom walls
    if (
      this.ball.y > this.height - this.ballRadius ||
      this.ball.y < this.ballRadius
    ) {
      this.ball.dy *= -1;
    }

    // Check if ball goes out of bounds (someone scores)
    if (this.ball.x < 0) {
      // Right player scores
      this.gameOver = true;
      return { winner: "right" };
    } else if (this.ball.x > this.width) {
      // Left player scores
      this.gameOver = true;
      return { winner: "left" };
    }

    return { winner: null };
  }

  // Get normalized inputs for neural network
  getInputs() {
    return [
      this.ball.x / this.width, // Ball X position
      this.ball.y / this.height, // Ball Y position
      this.ball.dx / (this.ballSpeedX * 2), // Ball X velocity
      this.ball.dy / (this.ballSpeedY * 2), // Ball Y velocity
      this.leftPaddle.y / (this.height - this.paddleHeight), // Left paddle position
      this.rightPaddle.y / (this.height - this.paddleHeight), // Right paddle position
    ];
  }
}

// Test the neural networks in a simulated Pong environment
function testGeneration(networks, maxFrames = 1000) {
  const results = [];

  // For each network in the population
  for (let i = 0; i < networks.length; i++) {
    const network = networks[i];

    // Create a new simulation for this network
    const sim = new PongSimulation();
    let frames = 0;
    let score = 0;

    // Run the simulation until game over or max frames reached
    while (!sim.gameOver && frames < maxFrames) {
      const inputs = sim.getInputs();

      // Get network output for right paddle (our AI)
      const rightPaddleMove = network.compute(inputs)[0];

      // Simple algorithm for left paddle (opponent)
      let leftPaddleMove;
      if (sim.ball.dx < 0) {
        // Ball is moving towards left paddle
        // Simple tracking: move towards the ball's y position
        const paddleCenter = sim.leftPaddle.y + sim.paddleHeight / 2;
        leftPaddleMove = paddleCenter > sim.ball.y ? 0 : 1;
      } else {
        // Random movement when ball is moving away
        leftPaddleMove = Math.random();
      }

      // Update simulation
      const result = sim.update(leftPaddleMove, rightPaddleMove);

      // Scoring: +1 for each frame survived, +10 for winning
      score += 1;

      if (result.winner === "right") {
        score += 10;
      }

      frames++;
    }

    results.push({ networkIndex: i, score });
    console.log(`Network ${i}: Score ${score}, Frames: ${frames}`);
  }

  return results;
}

// Main function to run the evolutionary process
function runEvolution(generations = 10) {
  console.log("Starting neuroevolution simulation...");

  // Initialize networks for first generation
  let networks = Neuro.nextGeneration();

  for (let gen = 1; gen <= generations; gen++) {
    console.log(`\n--- Generation ${gen} ---`);

    // Test all networks in this generation
    const results = testGeneration(networks);

    // Sort results by score (descending)
    results.sort((a, b) => b.score - a.score);

    // Report results
    console.log(
      `Best score: ${results[0].score} (Network ${results[0].networkIndex})`
    );
    console.log(
      `Average score: ${
        results.reduce((sum, r) => sum + r.score, 0) / results.length
      }`
    );

    // Submit scores to Neuro
    for (let i = 0; i < results.length; i++) {
      const { networkIndex, score } = results[i];
      Neuro.networkScore(networks[networkIndex], score);
    }

    // Generate next generation if not the last one
    if (gen < generations) {
      networks = Neuro.nextGeneration();
    }
  }

  console.log("\nEvolution complete!");

  // Return the final generation of networks
  return networks;
}

// Run the simulation for 10 generations
const trainedNetworks = runEvolution(10);

// Optional: Save the best network to a file
const bestNetwork = trainedNetworks[0];
const bestNetworkSave = bestNetwork.getSave();
fs.writeFileSync("best-network.json", JSON.stringify(bestNetworkSave, null, 2));
console.log("Best network saved to best-network.json");
