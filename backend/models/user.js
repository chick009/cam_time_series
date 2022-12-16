const mongoose = require('mongoose');
const AutoIncrement  = require('mongoose-sequence')(mongoose)
const uniqueValidator = require('mongoose-unique-validator');

const Schema = mongoose.Schema;

const userSchema = new Schema({
  name: { type: String, required: true },
  email: { type: String, required: true },
  password: { type: String, required: true, minlength: 3 },
  /*edit*/
  isAdmin: { type: Boolean, required: true, },
  /*edit*/
  image: { type: String, required: true },
  places: [{ type: mongoose.Types.ObjectId, required: true, ref: 'Place' }]
});

userSchema.plugin(AutoIncrement, {inc_field: 'userId'});
userSchema.plugin(uniqueValidator);

module.exports = mongoose.model('User', userSchema);
