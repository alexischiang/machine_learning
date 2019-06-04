import keras

a = keras.utils.to_categorical([3,1])
b = keras.utils.to_categorical([3,1],9)

print(a)
print(b)