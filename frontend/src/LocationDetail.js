
import React from 'react';
import Comment from './Comment';
import { useParams } from 'react-router-dom';
import Header from './Header';
import NavigationBar from './NavigationBar';
import Map from './Map';

const url = 'http://localhost';

function LocationDetail(props) {
  const [comment, setComment] = React.useState('');
  const [commentData, setCommentData] = React.useState([]);
  const [locationDetail, setLocationDetail] = React.useState({});
  const [isFavLoc, setIsFavLoc] = React.useState(false);

  let { locationName } = useParams();
  locationName = locationName.substring(1);

  // the location name has to change to props.location, to be handled when navigation is done
  const handleSubmit = async (event) => {
    event.preventDefault();
    let submission = document.getElementById('comment').value;
    if (submission === '') return;
    document.getElementById('comment').value = '';
    await setComment((comment) => (comment = submission));
    await fetch(url + '/postComment', {
      method: 'POST',
      body: JSON.stringify({
        location: locationName,
        username: props.username,
        content: submission,
      }),
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then((res) => console.log(res.success));
  };

  // fetch comment
  React.useEffect(() => {
    fetch(url + `/fetchComment/${locationName}`, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then((res) => {
        setCommentData((commentData) => (commentData = res));
      });
  }, [commentData, comment, locationName]);

  // fetch details of location
  React.useEffect(() => {
    fetch(url + `/fetchLocationDetails/${locationName}`, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then((res) => {
        setLocationDetail((locationDetail) => (locationDetail = res));
      });
  }, [locationName]);

  // fetch whether this location is user's favourite location
  React.useEffect(() => {
    fetch(url + `/checkFavLocation/${locationName}/${props.username}`, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then((res) => {
        if (res.isFavLoc === true) setIsFavLoc((isFavLoc) => (isFavLoc = true));
        else setIsFavLoc((isFavLoc) => (isFavLoc = false));
      });
  }, [locationName, props.username, isFavLoc]);

  function addFavLocation() {
    fetch(url + `/addFavLocation/${locationName}/${props.username}`, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then(() => {
        setIsFavLoc((isFavLoc) => (isFavLoc = true));
      });
  }

  function removeFavLocation() {
    fetch(url + `/delFavLocation/${locationName}/${props.username}`, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => res.json())
      .then(() => {
        setIsFavLoc((isFavLoc) => (isFavLoc = false));
      });
  }

  return (
    <div className="h-100 d-flex flex-column">
      <Header username={props.username}/>
      <NavigationBar />
      <Map filter={locationName} />
      <div className="d-flex my-2 ms-2 justify-content-between">
        <h1>{locationName}</h1>
        {isFavLoc ? (
          <button onClick={() => removeFavLocation()} className="btn btn-danger me-2">
            Remove from favourite
          </button>
        ) : (
          <button onClick={() => addFavLocation()} className="btn btn-success me-2">
            Add to favourite
          </button>
        )}
      </div>

      {/* <div className="d-flex w-100 align-items-center justify-content-center container"> */}
      <div className="container align-items-center justify-content-center w-100">

        <div className="row">

          <div id="detailSession" className="col-12 table-responsive text-nowrap" style={{ textAlign: 'left' }}>
            <h3>Details</h3>

            <table className="table table-responsive">
              <tr>
                {}
                <th scope="row">Latitude:</th>
                <td>{locationDetail.latitude}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Longitude:</th>
                <td>{locationDetail.longtitude}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Title </th>
                <td>{locationDetail.title}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Date/Time </th>
                <td>{locationDetail.data_time}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Description :</th>
                <td>{locationDetail.dscription}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Presenter :</th>
                <td>{locationDetail.presenter}</td>
              </tr>
              <tr>
                {}
                <th scope="row">Price {'($)'}:</th>
                <td>{locationDetail.price}</td>
              </tr>
            </table>

          </div>
        </div>

        <hr />

        <div className="w-100 row">
        <h3 style = {{textAlign: 'left'}}>Comments</h3>
          <div id="commentSession" className="col-12" style={{ textAlign: 'left' }}>
            
            {commentData !== null ? (
              commentData.map((item) => <Comment key={item.id} comment={item} />)
            ) : (
              <p>No comments found.</p>
            )}

          </div>
          <form className="pb-3" onSubmit={handleSubmit}>
              <div className="form-group">
                <textarea id="comment" className="form-control" type="text" placeholder="Comment..." style={{marginBottom: 10}}/>
              </div>
              <input type="submit" className="btn btn-info form-control" value="Post!" />
            </form>
        </div>
      </div>
    </div>
  );
}

export default LocationDetail;
