import ltn
import tensorflow as tf

@tf.function
def axioms(features, labels, training=False):
    x_A = ltn.Variable("x_A", features[labels == 0])
    x_B = ltn.Variable("x_B", features[labels == 1])
    x_C = ltn.Variable("x_C", features[labels == 2])
    x_D = ltn.Variable("x_D", features[labels == 3])
    x_E = ltn.Variable("x_E", features[labels == 4])
    x_F = ltn.Variable("x_F", features[labels == 5])
    x_G = ltn.Variable("x_G", features[labels == 6])
    x_H = ltn.Variable("x_H", features[labels == 7])
    x_I = ltn.Variable("x_I", features[labels == 8])
    axioms = [
        Forall(x_A, p([x_A, class_0], training=training)),
        Forall(x_B, p([x_B, class_1], training=training)),
        Forall(x_C, p([x_C, class_2], training=training)),
        Forall(x_D, p([x_D, class_3], training=training)),
        Forall(x_E, p([x_E, class_4], training=training)),
        Forall(x_F, p([x_F, class_5], training=training)),
        Forall(x_G, p([x_G, class_6], training=training)),
        Forall(x_H, p([x_H, class_7], training=training)),
        Forall(x_I, p([x_I, class_8], training=training))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step(features, labels):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels, training=True)
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    sat = axioms(features, labels) # compute sat without dropout
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = model([features])
    metrics_dict['train_accuracy'](tf.one_hot(labels,9),predictions)
    
@tf.function
def test_step(features, labels):
    # sat
    sat = axioms(features, labels)
    metrics_dict['test_sat_kb'](sat)
    # accuracy
    predictions = model([features])
    metrics_dict['test_accuracy'](tf.one_hot(labels,9),predictions)



