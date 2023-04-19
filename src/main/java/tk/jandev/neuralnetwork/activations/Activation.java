package tk.jandev.neuralnetwork.activations;

public enum Activation {

    SIGMOID {
        @Override
        public double calculate(double input) {
            return 1 / (1 + Math.exp(-input));
        }
    },

    IDENTITY {
        @Override
        public double calculate(double input) {
            return input;
        }
    },

    RELU {
        @Override
        public double calculate(double input) {
            if (input > 0) return input;
            return 0;
        }
    };



    public abstract double calculate(double input);
}
