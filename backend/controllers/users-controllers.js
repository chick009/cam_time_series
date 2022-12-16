const { validationResult } = require('express-validator');

const HttpError = require('../models/http-error');
const User = require('../models/user');

const getUsers = async (req, res, next) => {
  let users;
  try {
      users = await User.find({},);
  } catch (err) {
      console.log(users)
      const error = new HttpError(
          'getEvents failed, please try again later.',
          500
      );
      return next(error);
  }
  res.json({ users: users.map(user => user.toObject({ getters: true })) });
};


const getUser = async (req, res, next) => {
  const {id} = req.body;
  let user;
  try {
    user = await User.findOne({id}, '-password');
  } catch (err) {
    const error = new HttpError(
      'Fetching user failed, please try again later.',
      500
    );
    return next(error);
  }
  res.json({ user: user.toObject({ getters: true }) });
};

const signup = async (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return next(
      new HttpError('Invalid inputs passed, please check your data.', 422)
    );
  }
  const { name, password } = req.body;
/*
  let existingUser;
  try {
    existingUser = await User.findOne({ email: email });
  } catch (err) {
    const error = new HttpError(
      'Signing up failed, please try again later.',
      500
    );
    return next(error);
  }

  
  if (existingUser) {
    const error = new HttpError(
      'User exists already, please login instead.',
      422
    );
    return next(error);
  }*/
  const Admin_Boolean = (name == "admin")

  const createdUser = new User({
    name,
    email: "HAHA@gmail.com",
    image: 'https://live.staticflickr.com/7631/26849088292_36fc52ee90_b.jpg',
    password,
    isAdmin: Admin_Boolean,
    places: []
  });

  try {
    await createdUser.save();
  } catch (err) {
    const error = new HttpError(
      'Signing up failed, please try again later.',
      500
    );
    return next(error);
  }

  res.status(201).json({ user: createdUser.toObject({ getters: true }) });
};

const login = async (req, res, next) => {
  const { email, password } = req.body;

  let existingUser;

  try {
    existingUser = await User.findOne({ email: email });
  } catch (err) {
    const error = new HttpError(
      'Loggin in failed, please try again later.',
      500
    );
    return next(error);
  }

  if (!existingUser || existingUser.password !== password) {
    const error = new HttpError(
      'Invalid credentials, could not log you in.',
      401
    );
    return next(error);
  }

  res.json({
    message: 'Logged in!',
    user: existingUser.toObject({ getters: true })
  });
};

const deleteUser = async (req, res, next) => {
    const id = req.params['id'];
    let requireduser;
    
    try{
      requireduser = await User.deleteOne({ userId: id })
    }catch(err){
      const error = new HttpError(
        'Deletion failed, please try again later.',
        500
      );
      return next(error)
    }

}


 const updateUsers = async (req, res, next) => {
   let doc = req.body;
   const id = req.params['id'];
   try {
     doc = await User.findOneAndUpdate({userId: id}, doc), {
       new: true
     };
   } catch (err) {
     const error = new HttpError(
       'update Users failed, please try again later.',
       500
     );
     return next(error);
   }
   res.status(200).json({ doc });}
// };

exports.getUsers = getUsers;
exports.getUser = getUser;
exports.signup = signup;
exports.login = login;
exports.deleteUser = deleteUser;
exports.updateUsers = updateUsers;
